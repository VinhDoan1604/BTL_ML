import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import emoji
from collections import Counter
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def plot_label_distribution(df, label_column, label_order, colors, theme_palette):
    counts = df[label_column].value_counts().reindex(label_order).fillna(0)
    pcts = counts / counts.sum() * 100

    # ─── FIGURE SETUP ───────────────────────────────────────────────
    plt.style.use('default') # Clean white background
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)

    # ── 1. Count chart (SPANNING THE ENTIRE TOP ROW) ───────────────
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(label_order, counts, color=colors, width=0.4, edgecolor='white')

    # Calculate %
    total_count = counts.sum()
    offset_y1 = counts.max() * 0.03

    for bar, cnt in zip(bars, counts.values):
        pct = (cnt / total_count) * 100
        text_label = f'{cnt:,} ({pct:.1f}%)'

        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset_y1,
                 text_label, ha='center', va='bottom', fontsize=11)

    ax1.set_title('1. Count Distribution', fontweight='bold', fontsize=13, pad=10)
    ax1.set_xlabel('Rating Label')
    ax1.set_ylabel('Count')
    ax1.spines[['top','right']].set_visible(False)
    ax1.set_ylim(0, counts.max() * 1.2)

    # ── 2. Cumulative distribution (Ordinal specific) ───────────────
    ax2 = fig.add_subplot(gs[1, 0])
    cum = pcts.cumsum()
    main_color = colors[0] # Using the first color from the generated palette

    # Actual dataset line
    ax2.plot(label_order, cum.values, 'o-', color=main_color, lw=2.5,
             markersize=8, label='Actual (Dataset)')

    # Uniform distribution line (Ideal scenario: 20% each)
    uniform_cum = np.linspace(100/len(label_order), 100, len(label_order))
    ax2.plot(label_order, uniform_cum, '--', color='gray', lw=1.5, alpha=0.8, label='Ideal (Uniform)')

    # Shade the skewness area
    ax2.fill_between(label_order, cum.values, 0, alpha=0.15, color=main_color)

    ax2.set_title('2. Cumulative Distribution (Ordinal)', fontweight='bold', fontsize=13, pad=10)
    ax2.set_xlabel('Rating Label')
    ax2.set_ylabel('Cumulative %')
    ax2.set_ylim(0, 110)
    ax2.legend(loc='lower right')
    ax2.spines[['top','right']].set_visible(False)
    ax2.set_xticks(label_order)

    # ── 3. Imbalance Ratio (Rule of thumb: 3 and 10) ────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    max_count = counts.max()
    ir = max_count / counts

    # Plot IR bars using the consistent theme_palette
    bars_ir = ax3.bar(label_order, ir.values, color=colors, width=0.6, edgecolor='white')

    # Add Rule of Thumb baselines
    ax3.axhline(3, color='gray', ls='--', lw=1.5, label='IR = 3')
    ax3.axhline(10, color='black', ls='--', lw=1.5, label='IR = 10')

    offset_y4 = ir.max() * 0.05
    for i, (lbl, v) in enumerate(zip(label_order, ir.values)):
        # Alert text color if IR > 10
        text_color ='black'
        font_weight = 'bold' if v > 3 else 'normal'
        ax3.text(lbl, v + offset_y4, f'{v:.1f}×', ha='center', fontsize=11,
                 color=text_color, fontweight=font_weight)

    ax3.set_title('3. Imbalance Ratio (IR)', fontweight='bold', fontsize=13, pad=10)
    ax3.set_xlabel('Rating Label')
    ax3.set_ylabel('IR (Max Count / Count_i)')
    ax3.set_ylim(0, max(12, ir.max() * 1.2)) # Ensure the IR=10 line is always visible
    ax3.legend(loc='upper right', fontsize=10)
    ax3.spines[['top','right']].set_visible(False)

    # ─── SAVE AND DISPLAY ───────────────────────────────────────────
    print("Total  count: ", total_count)
    plt.tight_layout()
    # plt.savefig('label_distribution_eda.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_text_features(df, text_col='text'):
    # Create a copy to avoid Pandas SettingWithCopyWarning
    df_feat = df.copy()

    # Ensure the text column is string type and handle NaNs
    df_feat[text_col] = df_feat[text_col].fillna('').astype(str)

    # Word count (split by whitespace)
    df_feat['n_words'] = df_feat[text_col].apply(lambda x: len(str(x).split()))

    # Contains Emoji (Using emoji library for accuracy)
    df_feat['num_emoji'] = df_feat[text_col].apply(lambda x: emoji.emoji_count(x))

    # Uppercase Ratio - Very useful for sentiment/rating classification
    df_feat['upper_ratio']   = df_feat[text_col].apply(
    lambda x: sum(1 for c in x if c.isupper()) / len(x) if x else 0.0)

    print("Feature extraction completed successfully.")
    return df_feat

def plot_overall_text_features_distribution(df, text_feature_cols, theme_palette, limit_quantile=0.99):
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(text_feature_cols), 1, figsize=(20, 10 * len(text_feature_cols)))
    axes = np.atleast_1d(axes) # Ensure axes is always an array or list

    def plot_distribution(ax, data, title_prefix, x_label, y_label, limit_quantile, theme_palette):
        limit = data.quantile(limit_quantile)
        data_to_plot = data[data <= limit]

        # Create a histogram
        counts, bins, patches = ax.hist(
            data_to_plot,
            bins=50,
            edgecolor='white',
            linewidth=0.5
        )

        # Apply color gradient
        cmap = plt.get_cmap(theme_palette)
        x_min, x_max = bins[0], bins[-1]
        for p, x_val in zip(patches, bins[:-1]):
            bin_center = x_val + (bins[1] - bins[0]) / 2
            normalized_x = (bin_center - x_min) / (x_max - x_min)
            p.set_facecolor(cmap(normalized_x))

        # Add KDE plot
        ax2 = ax.twinx()
        sns.kdeplot(data=data_to_plot, color='gray', linewidth=2.5, ax=ax2)
        ax2.set_yticks([])
        ax2.set_ylabel('')

        # Set titles and labels
        ax.set_title(f'Overall {title_prefix} Distribution', fontweight='bold', fontsize=14, pad=15)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlim(0, limit)
        sns.despine(ax=ax, left=True)
        sns.despine(ax=ax2, left=True, right=True)

        # Add a small footnote to let the reader know the axis was capped
        ax.text(0.99, -0.05, '* X-axis capped at 99th percentile for visibility',
                transform=ax.transAxes, horizontalalignment='right',
                verticalalignment='top', fontsize=9, color='gray', style='italic')

    for i, feature_col in enumerate(text_feature_cols):
        title_prefix = feature_col.replace('n_', '').replace('_', ' ').title()
        x_label = f'Number of {title_prefix} per Review'
        y_label = 'Frequency (Count)'
        plot_distribution(
            axes[i],
            df[feature_col],
            title_prefix,
            x_label,
            y_label,
            limit_quantile,
            theme_palette
        )

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def plot_feature_distributions_by_label(df, feature_col, label_column, label_order, theme_palette):
    plt.style.use('default')
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f'Distribution of {feature_col.replace("n_", "").replace("_", " ").title()} by Rating Label', fontsize=18, fontweight='bold', y=1.05)

    colors_palette = sns.color_palette(theme_palette, len(label_order)).as_hex()

    # --- Subplot 1: KDE (Smoothed Histogram) ---
    ax_kde = axes[0]
    for label, color in zip(label_order, colors_palette):
        subset = df[df[label_column] == label]
        sns.kdeplot(
            data=subset,
            x=feature_col,
            fill=True,
            alpha=0.35,
            linewidth=2.5,
            color=color,
            label=f'Label {label}',
            ax=ax_kde
        )
    upper_limit_kde = df[feature_col].quantile(0.99)
    ax_kde.set_xlim(0, upper_limit_kde)
    ax_kde.set_title('KDE by Label', fontsize=14, fontweight='bold', pad=10)
    ax_kde.set_xlabel(feature_col.replace("n_", "").replace("_", " ").title(), fontsize=12)
    ax_kde.set_ylabel('Density', fontsize=12)
    ax_kde.legend(title='Rating Label')
    ax_kde.spines[['top', 'right']].set_visible(False)
    ax_kde.text(0.99, -0.1, '* X-axis capped at 99th percentile for visibility',
                transform=ax_kde.transAxes, horizontalalignment='right',
                verticalalignment='top', fontsize=9, color='gray', style='italic')

    # --- Subplot 2: Boxplot ---
    ax_boxplot = axes[1]
    sns.boxplot(
        data=df,
        x=label_column,
        y=feature_col,
        palette=colors_palette,
        hue=label_column,
        legend=False,
        width=0.5,
        linewidth=1.5,
        fliersize=3,
        flierprops={'marker': 'o', 'alpha': 0.5},
        ax=ax_boxplot
    )
    ax_boxplot.set_title('Boxplot by Label', fontsize=14, fontweight='bold', pad=10)
    ax_boxplot.set_xlabel('Rating Label', fontsize=12)
    ax_boxplot.set_ylabel(feature_col.replace("n_", "").replace("_", " ").title(), fontsize=12)
    ax_boxplot.spines[['top', 'right']].set_visible(False)

    # --- Subplot 3: ECDF (Cumulative Distribution Function) ---
    ax_ecdf = axes[2]
    for label, color in zip(label_order, colors_palette):
        subset = df[df[label_column] == label]
        sns.ecdfplot(
            data=subset,
            x=feature_col,
            color=color,
            linestyle='-',
            linewidth=2.5,
            label=f'Label {label}',
            ax=ax_ecdf
        )
    upper_limit_ecdf = df[feature_col].quantile(0.99)
    ax_ecdf.set_xlim(0, upper_limit_ecdf)
    ax_ecdf.set_title('ECDF by Label', fontsize=14, fontweight='bold', pad=10)
    ax_ecdf.set_xlabel(feature_col.replace("n_", "").replace("_", " ").title(), fontsize=12)
    ax_ecdf.set_ylabel('Cumulative Probability', fontsize=12)
    ax_ecdf.legend(title='Rating Label')
    ax_ecdf.spines[['top', 'right']].set_visible(False)
    ax_ecdf.text(0.99, -0.1, '* X-axis capped at 99th percentile for visibility',
                transform=ax_ecdf.transAxes, horizontalalignment='right',
                verticalalignment='top', fontsize=9, color='gray', style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent suptitle overlap
    plt.show()

def print_detailed_statistics_by_label(df, label_column, features):
    print("\n--- Detailed Statistics per Rating Label (n_words & n_chars) ---\n")

    for feature in features:
        print(f"--- {feature.replace('n_', '').replace('_', ' ').title()} Statistics ---")
        stats = df.groupby(label_column)[feature].agg(['mean', 'median', 'min', 'max', 'std']).round(2)
        print(stats.to_string())
        print("\n")

def plot_feature_trend_by_label(df, features, label_column, label_order, colors):
    # 1. Calculate Spearman correlation coefficients
    print("Spearman correlation with labels:")
    for col in features:
        r, p = spearmanr(df[col], df[label_column])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {col:20s}  r = {r:+.3f}  {sig}")

    # 2. Prepare data for plotting
    feat_means = df.groupby(label_column)[features].mean()

    # 3. Plotting
    fig, axes = plt.subplots(1, len(features), figsize=(18, 4))

    for i, (ax, col) in enumerate(zip(axes, features)):
        # Draw a line with color
        ax.plot(label_order, feat_means[col].values, 'o-',
                color=colors[i % len(colors)], markersize=8, lw=3, label=col)

        # Refine aesthetics
        ax.set_title(col, fontsize=12, fontweight='bold', color=colors[i % len(colors)])
        ax.set_xlabel('Rating (Label)', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3) # Add horizontal grid for readability

        # Remove top and right spines
        ax.spines[['top', 'right']].set_visible(False)

    # Overall title
    plt.suptitle('Feature Trend by Label',
                 fontweight='bold', fontsize=16, y=1.08, color='#333333')

    plt.tight_layout()
    plt.show()

def plot_emoji_analysis(df, text_column, label_column, label_order, colors):
    # --- 1. Prepare data for Visualization ---
    all_em = [char for text in df[text_column].dropna() for char in text if emoji.is_emoji(char)]

    has_emoji_count = (df['num_emoji'] > 0).sum()
    no_emoji_count = len(df) - has_emoji_count
    has_emoji_pct = (has_emoji_count / len(df)) * 100
    no_emoji_pct = 100 - has_emoji_pct

    top_12_em = Counter(all_em).most_common(12)
    em_names = [emoji.demojize(e) for e, _ in top_12_em]
    em_counts = [c for _, c in top_12_em]

    em_rate = df.groupby(label_column)['num_emoji'].apply(lambda x: (x > 0).mean()) * 100

    # --- 2. Initialize Canvas ---
    fig = plt.figure(figsize=(20, 20), dpi=100)
    gs = fig.add_gridspec(2, 2, hspace = 0.4, wspace = 0.3)

    # --- 3. LEFT: Donut Chart (Emoji Prevalence) ---
    ax1 = fig.add_subplot(gs[0,0])
    ax1.pie(
        [has_emoji_pct, no_emoji_pct],
        colors=[colors[0], '#D5D8DC'],
        startangle=90,
        counterclock=False,
        wedgeprops={'width': 0.45, 'edgecolor': 'w', 'linewidth': 5}
    )

    # Central KPI
    ax1.text(0, 0.05, f"{has_emoji_pct:.1f}%", ha='center', va='center',
                 fontsize=35, fontweight='bold', color=colors[0])
    ax1.text(0, -0.15, "Emoji Presence", ha='center', va='center',
                 fontsize=11, color='#2C3E50', fontweight='semibold', alpha=0.7)
    ax1.set_title('Non-Verbal Feature Distribution', fontsize=15, fontweight='bold', pad=20)

    # --- 4. TOP RIGHT: Bar Chart (% Reviews with Emoji by Rating) ---
    ax2 = fig.add_subplot(gs[0,1])
    ax2.bar(em_rate.index, em_rate.values,
                color=colors, width=0.6)
    ax2.set_title('% Reviews with Emoji by Rating', fontsize=15, fontweight='bold', pad=20)
    ax2.set_xlabel('Rating', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    for idx, val in em_rate.items():
        ax2.text(idx, val + 0.3, f'{val:.0f}%', ha='center', fontsize=9)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_overall_top_words(df, text_col, top_n=20, ngram_range=(1, 1), title="Unigrams", theme_palette="inferno"):
    """
    Vẽ biểu đồ tần suất từ vựng trên TOÀN BỘ tập dữ liệu (Global Bag of Words)
    """
    texts = df[text_col].dropna().tolist()

    # 1. Khởi tạo Vectorizer
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=10000)

    # 2. Fit và transform để đếm tần suất
    X_dtm = vectorizer.fit_transform(texts)

    # 3. Tính tổng tần suất của từng từ trên toàn corpus
    sum_words = np.asarray(X_dtm.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()

    # 4. Sắp xếp và lấy Top N
    sorted_idx = sum_words.argsort()[::-1][:top_n]
    top_words = [words[i] for i in sorted_idx]
    top_freq  = [sum_words[i] for i in sorted_idx]

    # 5. Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    
    # Đồng bộ palette truyền từ ngoài vào
    sns.barplot(x=top_freq, y=top_words, palette=theme_palette, edgecolor='white')

    plt.title(f"Top {top_n} Overall {title} in Corpus", fontweight='bold', fontsize=14, pad=15)
    plt.xlabel('Global Frequency', fontsize=12)
    plt.ylabel('N-gram', fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.show()

def get_ngram_description(ngram_range):
    """Generates a descriptive string for the n-gram range."""
    if ngram_range == (1, 1):
        return "Unigram"
    elif ngram_range == (2, 2):
        return "Bigram"
    elif ngram_range == (3, 3):
        return "Trigram"
    elif ngram_range == (1, 3):
        return "Unigram, Bigram, and Trigram"
    elif ngram_range == (1, 2):
        return "Unigram and Bigram"
    else:
        return f"N-gram ({ngram_range[0]}-{ngram_range[1]})"

def get_ngram_filename_suffix(ngram_range):
    """Generates a filename suffix for the n-gram range."""
    if ngram_range == (1, 1):
        return "uni"
    elif ngram_range == (2, 2):
        return "bi"
    elif ngram_range == (3, 3):
        return "tri"
    elif ngram_range == (1, 3):
        return "uni_bi_tri"
    elif ngram_range == (1, 2):
        return "uni_bi"
    else:
        return f"ngram_{ngram_range[0]}_{ngram_range[1]}"

# ĐÃ BỔ SUNG THAM SỐ `label_order` VÀ `colors` ĐỂ TRÁNH LỖI BIẾN TOÀN CỤC KHÔNG XÁC ĐỊNH
def plot_top_words_by_label(df, text_col, label_col, label_order, colors, top_n=15, ngram_range=(1,1)):
    # Thay vì tự sort unique, dùng label_order được truyền vào để đảm bảo thứ tự khớp màu
    n_labels = len(label_order)

    fig, axes = plt.subplots(1, n_labels, figsize=(4 * n_labels, 6), sharey=False)

    for ax, star, color in zip(axes, label_order, colors):
        texts = df[df[label_col] == star][text_col].dropna().tolist()

        vec = CountVectorizer(
            max_features=top_n,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2
        ).fit(texts)

        # Count frequency
        X = vec.transform(texts)
        freq = np.asarray(X.sum(axis=0)).flatten()
        words = vec.get_feature_names_out()

        # Sort
        sorted_idx = freq.argsort()[::-1]
        top_words = [words[i] for i in sorted_idx]
        top_freq  = [freq[i]  for i in sorted_idx]

        # Plot
        ax.barh(top_words[::-1], top_freq[::-1], color=color, edgecolor='white')
        ax.set_title(f'Label {star}', fontweight='bold', fontsize=13)
        ax.set_xlabel('Frequency')
        ax.tick_params(axis='y', labelsize=9)

    ngram_desc = get_ngram_description(ngram_range)
    filename_suffix = get_ngram_filename_suffix(ngram_range)

    plt.suptitle(f'Top {top_n} {ngram_desc} Words by Rating',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    # plt.savefig(f'eda_05_top_words_{filename_suffix}.png', bbox_inches='tight')
    plt.show()

def plot_tfidf_features_per_class(df, text_column, label_column, label_order, colors, theme_palette, max_features=8000, ngram_range=(1, 5)):
    label_docs = {
        lbl: ' '.join(df[df[label_column] == lbl][text_column].tolist())
        for lbl in label_order
    }

    vec_eda = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)
    X_eda   = vec_eda.fit_transform(label_docs.values())
    vocab   = vec_eda.get_feature_names_out()

    all_scores = X_eda.toarray()
    global_max = all_scores.max() * 1.1

    fig, axes = plt.subplots(len(label_order), 1, figsize=(20, 4 * len(label_order)), sharex=True)
    if len(label_order) == 1: # Handle case with a single subplot
        axes = [axes]

    for i, (ax, lbl, color) in enumerate(zip(axes, label_order, colors)):
        scores  = all_scores[i]
        top_idx = scores.argsort()[::-1][:12]

        ax.barh(vocab[top_idx][::-1], scores[top_idx][::-1],
                color=color, alpha=0.75, edgecolor='white')

        ax.set_xlim(0, global_max)

        ax.set_title(f'Label {lbl}', color=color, fontweight='bold')
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y', labelsize=9)

    plt.suptitle('Top TF-IDF n-gram features per class', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# Vocab overlap (Jaccard) between labels
def get_vocab_set(texts, top_k=500):
    counter = Counter(' '.join(texts).split())
    return set([w for w, _ in counter.most_common(top_k)])

def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0

def plot_jaccard_similarity_heatmap(df, label_column, label_order, theme_palette):
    vocab_per_label = {
        lbl: get_vocab_set(df[df[label_column] == lbl]['processed_text'])
        for lbl in label_order
    }

    jac = np.array([[jaccard(vocab_per_label[a], vocab_per_label[b])
                     for b in label_order] for a in label_order])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(jac, annot=True, fmt='.2f', cmap=theme_palette,
                xticklabels=label_order , yticklabels=label_order,
                linewidths=0.5, ax=ax)
    ax.set_title('Jaccard similarity of vocabulary between labels', fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("Adjacent labels have high Jaccard similarity → high adjacent confusion → should use ordinal-aware evaluation")