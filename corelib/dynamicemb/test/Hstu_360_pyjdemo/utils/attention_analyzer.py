import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
        
        # 取任意一层的 L 来计算实际 seq 长度
        first_attn = next(iter(item["attn"].values()))
        L = first_attn.shape[1]
        actual_seq_len = L - 1 - context_token_num - candidate_token_num  # 减去 profile + context + candidate
        
        if actual_seq_len < min_seq_len:
            print(f"[Visualize] Skipping sample {idx} (key={sample_key}): seq_len={actual_seq_len} < {min_seq_len}")
            continue
        
        for layer_name, attn_matrix in item["attn"].items():
            # attn_matrix: [num_heads, L, L]
            num_heads = attn_matrix.shape[0]
            seq_len = attn_matrix.shape[1]
            
            # ========== 图1: 每个 head 的 attention heatmap ==========
            fig, axes = plt.subplots(
                1, num_heads, 
                figsize=(6 * num_heads, 5),
                squeeze=False,
            )
            fig.suptitle(
                f"Sample: {sample_key} | Label: {sample_label} | pCTR: {sample_pctr:.4f} | seq_len: {actual_seq_len}\n{layer_name}",
                fontsize=14,
            )
            
            for h in range(num_heads):
                ax = axes[0, h]
                im = ax.imshow(
                    attn_matrix[h], 
                    cmap="viridis", 
                    aspect="auto",
                    vmin=0,
                    vmax=attn_matrix[h].max(),
                )
                ax.set_title(f"Head {h}", fontsize=11)
                ax.set_xlabel("Key position")
                ax.set_ylabel("Query position")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            fname = os.path.join(save_dir, f"sample_{idx}_{layer_name}_heads.png")
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # ========== 图2: 平均 attention (所有 head 取均值) ==========
            avg_attn = attn_matrix.mean(axis=0)  # [L, L]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(avg_attn, cmap="viridis", aspect="auto")
            ax.set_title(
                f"Avg Attention | {sample_key} | Label={sample_label} pCTR={sample_pctr:.4f} | seq_len: {actual_seq_len}\n{layer_name}",
                fontsize=12,
            )
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            fig.colorbar(im, ax=ax)
            
            # 标注 token 区域 —— 传入实际的 token 数量
            _add_region_annotations(ax, seq_len, candidate_token_num, context_token_num)
            
            plt.tight_layout()
            fname = os.path.join(save_dir, f"sample_{idx}_{layer_name}_avg.png")
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # ========== 图3: Candidate tokens 对 Sequence tokens 的 attention 分布 ==========
            _plot_candidate_to_seq_attention(
                attn_matrix, idx, layer_name, sample_key, 
                sample_label, sample_pctr, save_dir,
                candidate_token_num=candidate_token_num,
                context_token_num=context_token_num,
                actual_seq_len=actual_seq_len,
            )

            # ========== 图4: 分区域注意力占比 (饼图 + 条形图) ==========
            _plot_region_attention_summary(
                attn_matrix, idx, layer_name, sample_key,
                sample_label, sample_pctr, save_dir,
                candidate_token_num=candidate_token_num,
                context_token_num=context_token_num,
                actual_seq_len=actual_seq_len,
            )
        
        plotted += 1
    
    print(f"[Visualize] Saved {plotted} attention samples to {save_dir}")


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
    actual_seq_len=0,
):
    """
    绘制 candidate tokens 对所有 position 的 attention 分布（bar chart）。
    attn_matrix: [H, L, L]
    """
    num_heads, L, _ = attn_matrix.shape
    avg_attn = attn_matrix.mean(axis=0)  # [L, L]
    
    # token 布局: [1 profile] [seq_len seq] [context_token_num context] [candidate_token_num candidate]
    candidate_start = L - candidate_token_num
    context_start = candidate_start - context_token_num
    profile_end = 1  # profile token 固定在第一个位置
    # candidate tokens 对所有位置的平均 attention 权重（先对 head 取平均，再对 candidate tokens 取平均）
    cand_to_all = avg_attn[candidate_start:, :].mean(axis=0)

    # ---- 调试打印 ----
    print(f"[DEBUG] sample={sample_idx} {layer_name} actual_seq_len={actual_seq_len}")
    print(f"  L={L}, profile_end={profile_end}, context_start={context_start}, candidate_start={candidate_start}")
    print(f"  cand→profile  : {cand_to_all[:profile_end].sum():.6f}")
    print(f"  cand→sequence : {cand_to_all[profile_end:context_start].sum():.6f}")
    print(f"  cand→context  : {cand_to_all[context_start:candidate_start].sum():.6f}")
    print(f"  cand→candidate: {cand_to_all[candidate_start:].sum():.6f}")

    colors = _get_region_colors(L, profile_end, context_start, candidate_start)
    positions = np.arange(L)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(min(24, L * 0.15 + 4), 8), sharex=True)
    fig.suptitle(
        f"Candidate → All | {sample_key} | Label={sample_label} pCTR={sample_pctr:.4f} | seq_len={actual_seq_len}\n{layer_name}",
        fontsize=12,
    )

    # 线性
    ax1.bar(positions, cand_to_all, color=colors, alpha=0.8, width=0.8)
    ax1.set_ylabel("Attention Weight (linear)")
    _add_bar_region_lines(ax1, profile_end, context_start, candidate_start)

    # log
    cand_log = np.where(cand_to_all > 0, cand_to_all, 1e-10)
    ax2.bar(positions, cand_log, color=colors, alpha=0.8, width=0.8)
    ax2.set_yscale('log')
    ax2.set_ylabel("Attention Weight (log)")
    ax2.set_xlabel("Token Position")
    _add_bar_region_lines(ax2, profile_end, context_start, candidate_start)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Profile'),
        Patch(facecolor='#3498db', label='Sequence'),
        Patch(facecolor='#2ecc71', label='Context'),
        Patch(facecolor='#f39c12', label='Candidate (self)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"sample_{sample_idx}_{layer_name}_cand2all_bar.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_region_attention_summary(
    attn_matrix, sample_idx, layer_name, sample_key,
    sample_label, sample_pctr, save_dir,
    candidate_token_num=8,
    context_token_num=1,
    actual_seq_len=0,
):
    """饼图 + 水平条形图：分区域注意力占比。"""
    num_heads, L, _ = attn_matrix.shape
    avg_attn = attn_matrix.mean(axis=0)

    candidate_start = L - candidate_token_num
    context_start = candidate_start - context_token_num
    profile_end = 1

    cand_to_all = avg_attn[candidate_start:, :].mean(axis=0)

    regions = {
        'Profile':   cand_to_all[:profile_end].sum(),
        'Sequence':  cand_to_all[profile_end:context_start].sum(),
        'Context':   cand_to_all[context_start:candidate_start].sum(),
        'Candidate': cand_to_all[candidate_start:].sum(),
    }
    region_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Region Summary | {sample_key} | Label={sample_label} pCTR={sample_pctr:.4f} | seq_len={actual_seq_len}\n{layer_name}",
        fontsize=11,
    )

    values = list(regions.values())
    labels_pie = [f"{k}\n{v:.4f}" for k, v in regions.items()]
    ax1.pie(values, labels=labels_pie, colors=region_colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Pie", fontsize=10)

    y_pos = np.arange(len(regions))
    ax2.barh(y_pos, values, color=region_colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(list(regions.keys()))
    ax2.set_xlabel("Total Attention Weight")
    ax2.set_title("Bar", fontsize=10)
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