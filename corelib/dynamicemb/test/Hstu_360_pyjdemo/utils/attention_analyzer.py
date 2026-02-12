import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _detect_real_seq_len(attn_matrix, candidate_token_num=8, context_token_num=1, threshold=1e-6):
    """
    根据 attention weights 检测真实序列长度（排除 padding）。
    
    逻辑：
    1. 对所有 head 取均值 → [L, L]
    2. 对所有 query 取均值 → [L]（每个 key position 被关注的总量）
    3. 在 seq 区域（排除 profile/context/candidate）中，
       从后往前找最后一个 > threshold 的位置 → 真实 seq 结束位置
    
    Args:
        attn_matrix: [H, L, L]
        threshold: 低于此阈值视为 padding
    
    Returns:
        real_seq_len: 真实序列长度（不含 profile/context/candidate）
    """
    H, L, _ = attn_matrix.shape
    
    candidate_start = L - candidate_token_num
    context_start = candidate_start - context_token_num
    profile_end = 1
    
    # 方法：看 candidate tokens（作为 query）对 seq 区域每个位置的注意力
    # candidate 行对 seq 区域的注意力能反映哪些位置是有效的
    avg_attn = attn_matrix.mean(axis=0)  # [L, L]
    cand_to_seq = avg_attn[candidate_start:, profile_end:context_start]  # [cand_num, seq_padded_len]
    
    # 对 candidate tokens 取均值 → [seq_padded_len]
    seq_attention = cand_to_seq.mean(axis=0)
    
    # 从后往前找最后一个非零位置
    real_seq_len = 0
    for i in range(len(seq_attention) - 1, -1, -1):
        if seq_attention[i] > threshold:
            real_seq_len = i + 1
            break
    
    return real_seq_len


def _get_effective_positions(L, real_seq_len, candidate_token_num=8, context_token_num=1):
    """
    返回有效 token 的位置索引和区域标签。
    
    原始布局: [1 profile] [real_seq] [padding...] [context] [candidate]
    有效布局: [1 profile] [real_seq] [context] [candidate]  (去掉 padding)
    
    Returns:
        effective_indices: 有效位置在原始 L 中的索引
        region_labels: 每个有效位置的区域标签 ('P', 'S', 'Ctx', 'C')
    """
    profile_end = 1
    candidate_start = L - candidate_token_num
    context_start = candidate_start - context_token_num
    
    indices = []
    labels = []
    
    # Profile
    for i in range(profile_end):
        indices.append(i)
        labels.append('P')
    
    # Real sequence (skip padding)
    for i in range(profile_end, profile_end + real_seq_len):
        indices.append(i)
        labels.append('S')
    
    # Context
    for i in range(context_start, candidate_start):
        indices.append(i)
        labels.append('Ctx')
    
    # Candidate
    for i in range(candidate_start, L):
        indices.append(i)
        labels.append('C')
    
    return np.array(indices), labels


def attention_analyzer(collected_attn, save_dir, max_samples=5, candidate_token_num=8, context_token_num=1, min_seq_len=5):
    """
    可视化 attention weights 并保存为图片。
    
    Args:
        collected_attn: List of dict, 每个 dict 包含:
            - step, key, label, pctr
            - attn: {layer_0: [H, L, L], layer_1: [H, L, L], ...}
        save_dir: 保存图片的目录
        max_samples: 最多可视化几个样本
        min_seq_len: 最小序列长度，低于此长度的样本跳过可视化
    """
    
    plotted = 0
    for idx, item in enumerate(collected_attn):
        if plotted >= max_samples:
            break

        sample_key = item["key"]
        sample_label = item["label"]
        sample_pctr = item["pctr"]
        
        # 用第一层 attention 检测真实 seq 长度
        first_attn = next(iter(item["attn"].values()))
        L = first_attn.shape[1]
        real_seq_len = _detect_real_seq_len(
            first_attn, candidate_token_num, context_token_num
        )
        
        if real_seq_len < min_seq_len:
            print(f"[Visualize] Skipping sample {idx} (key={sample_key}): real_seq_len={real_seq_len} < {min_seq_len}")
            continue
        
        print(f"[Visualize] Sample {idx} (key={sample_key}): L={L}, real_seq_len={real_seq_len}")
        
        for layer_name, attn_matrix in item["attn"].items():
            # attn_matrix: [num_heads, L, L]
            num_heads = attn_matrix.shape[0]
            
            # 获取有效位置
            eff_indices, eff_labels = _get_effective_positions(
                L, real_seq_len, candidate_token_num, context_token_num
            )
            eff_len = len(eff_indices)
            
            # 提取有效子矩阵: [H, eff_len, eff_len]
            attn_effective = attn_matrix[:, eff_indices][:, :, eff_indices]
            
            # ========== 图1: 每个 head 的 attention heatmap ==========
            fig, axes = plt.subplots(
                1, num_heads, 
                figsize=(6 * num_heads, 5),
                squeeze=False,
            )
            fig.suptitle(
                f"Sample: {sample_key} | Label: {sample_label} | pCTR: {sample_pctr:.4f}\n"
                f"seq_len: {real_seq_len} | total_L: {L} | {layer_name}",
                fontsize=14,
            )
            
            # tick labels
            tick_labels = []
            for i, (orig_idx, lbl) in enumerate(zip(eff_indices, eff_labels)):
                tick_labels.append(f"{lbl}{orig_idx}")
            
            for h in range(num_heads):
                ax = axes[0, h]
                im = ax.imshow(
                    attn_effective[h], 
                    cmap="viridis", 
                    aspect="auto",
                    vmin=0,
                    vmax=attn_effective[h].max(),
                )
                ax.set_title(f"Head {h}", fontsize=11)
                ax.set_xlabel("Key position")
                ax.set_ylabel("Query position")
                
                # 当有效 token 数不太多时，显示 tick labels
                if eff_len <= 30:
                    ax.set_xticks(range(eff_len))
                    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
                    ax.set_yticks(range(eff_len))
                    ax.set_yticklabels(tick_labels, fontsize=6)
                
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            fname = os.path.join(save_dir, f"sample_{idx}_{layer_name}_heads.png")
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # ========== 图2: 平均 attention (有效部分) ==========
            avg_attn_eff = attn_effective.mean(axis=0)  # [eff_len, eff_len]
            
            fig, ax = plt.subplots(figsize=(max(6, eff_len * 0.3), max(5, eff_len * 0.25)))
            im = ax.imshow(avg_attn_eff, cmap="viridis", aspect="auto")
            ax.set_title(
                f"Avg Attention (effective only) | {sample_key}\n"
                f"Label={sample_label} pCTR={sample_pctr:.4f} | real_seq={real_seq_len} | {layer_name}",
                fontsize=11,
            )
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            fig.colorbar(im, ax=ax)
            
            # 有效子矩阵中的区域分界
            _add_effective_region_annotations(ax, eff_labels)
            
            if eff_len <= 30:
                ax.set_xticks(range(eff_len))
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
                ax.set_yticks(range(eff_len))
                ax.set_yticklabels(tick_labels, fontsize=6)
            
            plt.tight_layout()
            fname = os.path.join(save_dir, f"sample_{idx}_{layer_name}_avg.png")
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # ========== 图3: Candidate → All 柱状图 (只看有效位置) ==========
            _plot_candidate_to_seq_attention(
                attn_matrix, idx, layer_name, sample_key, 
                sample_label, sample_pctr, save_dir,
                candidate_token_num=candidate_token_num,
                context_token_num=context_token_num,
                real_seq_len=real_seq_len,
            )

            # ========== 图4: 分区域注意力占比 (饼图 + 条形图) ==========
            _plot_region_attention_summary(
                attn_matrix, idx, layer_name, sample_key,
                sample_label, sample_pctr, save_dir,
                candidate_token_num=candidate_token_num,
                context_token_num=context_token_num,
                real_seq_len=real_seq_len,
            )
        
        plotted += 1
    
    print(f"[Visualize] Saved {plotted} attention samples to {save_dir}")


def _add_effective_region_annotations(ax, eff_labels):
    """在有效子矩阵的 heatmap 上画区域分界线。"""
    # 找区域边界
    prev_label = eff_labels[0]
    boundary_colors = {'P': '#e74c3c', 'S': '#3498db', 'Ctx': '#2ecc71', 'C': '#f39c12'}
    label_names = {'P': 'Profile', 'S': 'Sequence', 'Ctx': 'Context', 'C': 'Candidate'}
    
    for i in range(1, len(eff_labels)):
        if eff_labels[i] != prev_label:
            pos = i - 0.5
            color = boundary_colors.get(eff_labels[i], 'gray')
            name = f"{label_names.get(prev_label, '?')}|{label_names.get(eff_labels[i], '?')}"
            ax.axhline(y=pos, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axvline(x=pos, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            ax.text(len(eff_labels) + 0.3, pos, name, color=color, fontsize=7,
                    va='center', ha='left', fontweight='bold')
            prev_label = eff_labels[i]


def _add_region_annotations(ax, total_len, candidate_token_num=8, context_token_num=1):
    """在 attention heatmap 上添加 token 区域分界线。
    
    Token 布局: [1 profile] [L seq] [context_token_num context] [candidate_token_num candidate]
    """
    profile_end = 1
    candidate_start = total_len - candidate_token_num
    context_start = candidate_start - context_token_num
    
    # 画分界线（水平 + 垂直）
    boundaries = [
        (profile_end - 0.5, 'P|S', '#e74c3c'),  # Profile|Seq
        (context_start - 0.5, 'S|Ctx', '#2ecc71'),    # Seq|Ctx
        (candidate_start - 0.5, 'Ctx|C', '#f39c12'), # Ctx|Cand
    ]
    
    for pos, label, color in boundaries:
        if pos > 0 and pos < total_len:
            ax.axhline(y=pos, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axvline(x=pos, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            # 在右侧标注区域名
            ax.text(total_len + 0.5, pos, label, color=color, fontsize=7,
                    va='center', ha='left', fontweight='bold')


def _plot_candidate_to_seq_attention(
    attn_matrix, sample_idx, layer_name, sample_key,
    sample_label, sample_pctr, save_dir,
    candidate_token_num=8,
    context_token_num=1,
    real_seq_len=0,
):
    """
    绘制 candidate tokens 对有效 position 的 attention 分布（去掉 padding）。
    """
    num_heads, L, _ = attn_matrix.shape
    avg_attn = attn_matrix.mean(axis=0)  # [L, L]
    
    # token 布局: [1 profile] [seq_len seq] [context_token_num context] [candidate_token_num candidate]
    candidate_start = L - candidate_token_num
    context_start = candidate_start - context_token_num
    profile_end = 1
    
    # candidate → all 的原始权重 [L]
    cand_to_all_full = avg_attn[candidate_start:, :].mean(axis=0)
    
    # 只提取有效位置的权重
    eff_indices, eff_labels = _get_effective_positions(
        L, real_seq_len, candidate_token_num, context_token_num
    )
    cand_to_eff = cand_to_all_full[eff_indices]  # [eff_len]
    eff_len = len(eff_indices)
    
    # 分区域统计
    region_sums = {'Profile': 0.0, 'Sequence': 0.0, 'Context': 0.0, 'Candidate': 0.0}
    label_to_region = {'P': 'Profile', 'S': 'Sequence', 'Ctx': 'Context', 'C': 'Candidate'}
    for val, lbl in zip(cand_to_eff, eff_labels):
        region_sums[label_to_region[lbl]] += val
    
    # padding 部分被丢弃的权重
    padding_weight = cand_to_all_full.sum() - cand_to_eff.sum()
    
    # ---- 调试打印 ----
    print(f"[DEBUG] sample={sample_idx} {layer_name} real_seq_len={real_seq_len}")
    print(f"  L={L}, eff_len={eff_len}, padding_positions={L - eff_len}")
    for k, v in region_sums.items():
        print(f"  cand→{k:10s}: {v:.6f}")
    print(f"  cand→padding  : {padding_weight:.6f}")
    print(f"  total (w/ pad) : {cand_to_all_full.sum():.6f}")
    
    # ---- 颜色 ----
    color_map = {'P': '#e74c3c', 'S': '#3498db', 'Ctx': '#2ecc71', 'C': '#f39c12'}
    colors = [color_map[lbl] for lbl in eff_labels]
    
    # ---- 双子图 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(min(24, eff_len * 0.4 + 4), 8), sharex=True)
    fig.suptitle(
        f"Candidate → Effective Positions (padding removed)\n"
        f"{sample_key} | Label={sample_label} pCTR={sample_pctr:.4f} | "
        f"real_seq={real_seq_len} | {layer_name}",
        fontsize=11,
    )
    
    positions = np.arange(eff_len)
    
    # tick labels
    tick_labels = [f"{lbl}{orig}" for orig, lbl in zip(eff_indices, eff_labels)]
    
    # 线性
    ax1.bar(positions, cand_to_eff, color=colors, alpha=0.8, width=0.8)
    ax1.set_ylabel("Attention Weight (linear)")
    ax1.set_title("Linear Scale", fontsize=10)
    if eff_len <= 40:
        ax1.set_xticks(positions)
        ax1.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    # 在柱子上标数值（当 token 数不多时）
    if eff_len <= 25:
        for i, v in enumerate(cand_to_eff):
            if v > 0.001:
                ax1.text(i, v + 0.002, f"{v:.3f}", ha='center', va='bottom', fontsize=6)
    
    # 画区域分界线
    _add_effective_bar_region_lines(ax1, eff_labels)
    
    # log
    cand_log = np.where(cand_to_eff > 0, cand_to_eff, 1e-10)
    ax2.bar(positions, cand_log, color=colors, alpha=0.8, width=0.8)
    ax2.set_yscale('log')
    ax2.set_ylabel("Attention Weight (log)")
    ax2.set_xlabel("Effective Token Position")
    ax2.set_title("Log Scale", fontsize=10)
    if eff_len <= 40:
        ax2.set_xticks(positions)
        ax2.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    _add_effective_bar_region_lines(ax2, eff_labels)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label=f"Profile ({region_sums['Profile']:.4f})"),
        Patch(facecolor='#3498db', label=f"Sequence ({region_sums['Sequence']:.4f})"),
        Patch(facecolor='#2ecc71', label=f"Context ({region_sums['Context']:.4f})"),
        Patch(facecolor='#f39c12', label=f"Candidate ({region_sums['Candidate']:.4f})"),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    fname = os.path.join(save_dir, f"sample_{sample_idx}_{layer_name}_cand2all_bar.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_region_attention_summary(
    attn_matrix, sample_idx, layer_name, sample_key,
    sample_label, sample_pctr, save_dir,
    candidate_token_num=8,
    context_token_num=1,
    real_seq_len=0,
):
    """饼图 + 水平条形图：分区域注意力占比（只看有效位置）。"""
    num_heads, L, _ = attn_matrix.shape
    avg_attn = attn_matrix.mean(axis=0)

    candidate_start = L - candidate_token_num
    context_start = candidate_start - context_token_num
    profile_end = 1

    cand_to_all_full = avg_attn[candidate_start:, :].mean(axis=0)
    
    eff_indices, eff_labels = _get_effective_positions(
        L, real_seq_len, candidate_token_num, context_token_num
    )
    cand_to_eff = cand_to_all_full[eff_indices]
    
    label_to_region = {'P': 'Profile', 'S': 'Sequence', 'Ctx': 'Context', 'C': 'Candidate'}
    regions = {'Profile': 0.0, 'Sequence': 0.0, 'Context': 0.0, 'Candidate': 0.0}
    for val, lbl in zip(cand_to_eff, eff_labels):
        regions[label_to_region[lbl]] += val
    
    # padding 丢弃的权重
    padding_weight = cand_to_all_full.sum() - cand_to_eff.sum()
    if padding_weight > 1e-6:
        regions['Padding (leaked)'] = padding_weight

    region_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    if 'Padding (leaked)' in regions:
        region_colors.append('#cccccc')

    values = list(regions.values())
    
    if sum(values) < 1e-12:
        print(f"[Visualize] Skipping region summary for sample {sample_idx}: all weights ~0")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Region Summary (effective only) | {sample_key}\n"
        f"Label={sample_label} pCTR={sample_pctr:.4f} | real_seq={real_seq_len} | {layer_name}",
        fontsize=11,
    )

    labels_pie = [f"{k}\n{v:.4f}" for k, v in regions.items()]
    ax1.pie(values, labels=labels_pie, colors=region_colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Attention Distribution (Pie)", fontsize=10)

    y_pos = np.arange(len(regions))
    ax2.barh(y_pos, values, color=region_colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(list(regions.keys()))
    ax2.set_xlabel("Total Attention Weight")
    ax2.set_title("Attention Distribution (Bar)", fontsize=10)
    for i, v in enumerate(values):
        ax2.text(v + 0.001, i, f"{v:.4f}", va='center', fontsize=9)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"sample_{sample_idx}_{layer_name}_region_summary.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ================ 工具函数 ================

def _get_region_colors(L, profile_end, context_start, candidate_start):
    colors = []
    for p in range(L):
        if p < profile_end:
            colors.append('#e74c3c')
        elif p < context_start:
            colors.append('#3498db')
        elif p < candidate_start:
            colors.append('#2ecc71')
        else:
            colors.append('#f39c12')
    return colors


def _add_bar_region_lines(ax, profile_end, context_start, candidate_start):
    for pos in [profile_end - 0.5, context_start - 0.5, candidate_start - 0.5]:
        ax.axvline(x=pos, color='gray', linestyle=':', alpha=0.5, linewidth=1)


def _add_effective_bar_region_lines(ax, eff_labels):
    """在有效位置柱状图上画区域分界线。"""
    prev = eff_labels[0]
    for i in range(1, len(eff_labels)):
        if eff_labels[i] != prev:
            ax.axvline(x=i - 0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1)
            prev = eff_labels[i]