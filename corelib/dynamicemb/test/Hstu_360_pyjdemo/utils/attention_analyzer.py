import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def attention_analyzer(collected_attn, save_dir, max_samples=5, candidate_token_num=8, context_token_num=1):
    """
    可视化 attention weights 并保存为图片。
    
    Args:
        collected_attn: List of dict, 每个 dict 包含:
            - step, key, label, pctr
            - attn: {layer_0: [H, L, L], layer_1: [H, L, L], ...}
        save_dir: 保存图片的目录
        max_samples: 最多可视化几个样本
    """
    
    for idx, item in enumerate(collected_attn[:max_samples]):
        num_layers = len(item["attn"])
        sample_key = item["key"]
        sample_label = item["label"]
        sample_pctr = item["pctr"]
        
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
                f"Sample: {sample_key} | Label: {sample_label} | pCTR: {sample_pctr:.4f}\n{layer_name}",
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
                f"Avg Attention | {sample_key} | Label={sample_label} pCTR={sample_pctr:.4f}\n{layer_name}",
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
            )
    
    print(f"[Visualize] Saved attention plots to {save_dir}")


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
    profile_end = 1  # profile 只有 1 个 token
    
    # candidate tokens 对所有 positions 的平均注意力
    cand_to_all = avg_attn[candidate_start:, :].mean(axis=0)  # [L]
    
    fig, ax = plt.subplots(figsize=(min(20, L * 0.15 + 4), 4))
    positions = np.arange(L)
    
    # 颜色编码不同区域
    colors = []
    for p in positions:
        if p < profile_end:
            colors.append('#e74c3c')      # profile: 红
        elif p < context_start:
            colors.append('#3498db')      # sequence: 蓝
        elif p < candidate_start:
            colors.append('#2ecc71')      # context: 绿
        else:
            colors.append('#f39c12')      # candidate: 橙
    
    ax.bar(positions, cand_to_all, color=colors, alpha=0.8, width=0.8)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Avg Attention Weight")
    ax.set_title(
        f"Candidate → All Positions | {sample_key} | "
        f"Label={sample_label} pCTR={sample_pctr:.4f}\n{layer_name}",
        fontsize=11,
    )
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Profile'),
        Patch(facecolor='#3498db', label='Sequence'),
        Patch(facecolor='#2ecc71', label='Context'),
        Patch(facecolor='#f39c12', label='Candidate (self)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    fname = os.path.join(save_dir, f"sample_{sample_idx}_{layer_name}_cand2all_bar.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)