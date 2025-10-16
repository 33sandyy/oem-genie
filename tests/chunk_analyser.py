import json
import statistics
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

class ChunkAnalyzer:
    def __init__(self, chunks_file: str):
        """Initialize analyzer with chunks JSON file."""
        self.chunks_file = chunks_file
        self.chunks = self.load_chunks()
    
    def load_chunks(self):
        """Load chunks from JSON file."""
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_chunk_lengths(self):
        """Analyze chunk lengths and provide detailed statistics."""
        lengths = [len(chunk['text']) for chunk in self.chunks]
        word_counts = [len(chunk['text'].split()) for chunk in self.chunks]
        
        # Calculate statistics
        stats = {
            'total_chunks': len(self.chunks),
            'character_length': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': statistics.mean(lengths),
                'median': statistics.median(lengths),
                'stdev': statistics.stdev(lengths) if len(lengths) > 1 else 0
            },
            'word_count': {
                'min': min(word_counts),
                'max': max(word_counts),
                'mean': statistics.mean(word_counts),
                'median': statistics.median(word_counts),
                'stdev': statistics.stdev(word_counts) if len(word_counts) > 1 else 0
            }
        }
        
        return stats, lengths, word_counts
    
    def analyze_by_source(self):
        """Analyze chunk lengths grouped by source file."""
        by_source = defaultdict(list)
        
        for chunk in self.chunks:
            source = chunk.get('source', 'unknown')
            by_source[source].append(len(chunk['text']))
        
        source_stats = {}
        for source, lengths in by_source.items():
            source_stats[source] = {
                'count': len(lengths),
                'min': min(lengths),
                'max': max(lengths),
                'mean': statistics.mean(lengths),
                'median': statistics.median(lengths),
                'stdev': statistics.stdev(lengths) if len(lengths) > 1 else 0
            }
        
        return source_stats
    
    def analyze_by_page(self):
        """Analyze chunks per page."""
        by_page = defaultdict(lambda: {'chunks': 0, 'lengths': []})
        
        for chunk in self.chunks:
            source = chunk.get('source', 'unknown')
            page = chunk.get('page_number', 0)
            key = f"{source}_p{page}"
            by_page[key]['chunks'] += 1
            by_page[key]['lengths'].append(len(chunk['text']))
        
        # Calculate average chunks per page
        chunks_per_page = [info['chunks'] for info in by_page.values()]
        
        return by_page, {
            'avg_chunks_per_page': statistics.mean(chunks_per_page),
            'min_chunks_per_page': min(chunks_per_page),
            'max_chunks_per_page': max(chunks_per_page)
        }
    
    def check_uniformity(self, tolerance_percent=20):
        """
        Check if chunks are uniform within tolerance.
        Returns chunks that deviate significantly from the mean.
        """
        lengths = [len(chunk['text']) for chunk in self.chunks]
        mean_length = statistics.mean(lengths)
        tolerance = mean_length * (tolerance_percent / 100)
        
        outliers = []
        for chunk in self.chunks:
            chunk_len = len(chunk['text'])
            if abs(chunk_len - mean_length) > tolerance:
                outliers.append({
                    'id': chunk['id'],
                    'source': chunk.get('source', 'N/A'),
                    'page': chunk.get('page_number', 'N/A'),
                    'length': chunk_len,
                    'deviation': chunk_len - mean_length,
                    'deviation_percent': ((chunk_len - mean_length) / mean_length) * 100
                })
        
        return outliers, mean_length, tolerance
    
    def visualize_distribution(self, save_path='chunk_length_distribution.png'):
        """Create visualization of chunk length distribution."""
        lengths = [len(chunk['text']) for chunk in self.chunks]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Chunk Length Analysis', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0, 0].hist(lengths, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Chunk Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Chunk Lengths')
        axes[0, 0].axvline(statistics.mean(lengths), color='red', linestyle='--', 
                          label=f'Mean: {statistics.mean(lengths):.0f}')
        axes[0, 0].legend()
        
        # Box plot
        axes[0, 1].boxplot(lengths, vert=True)
        axes[0, 1].set_ylabel('Chunk Length (characters)')
        axes[0, 1].set_title('Box Plot of Chunk Lengths')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Length vs Chunk Index (shows variation over document)
        axes[1, 0].plot(range(len(lengths)), lengths, alpha=0.6)
        axes[1, 0].axhline(statistics.mean(lengths), color='red', linestyle='--', 
                          label='Mean')
        axes[1, 0].set_xlabel('Chunk Index')
        axes[1, 0].set_ylabel('Chunk Length (characters)')
        axes[1, 0].set_title('Chunk Length Over Document')
        axes[1, 0].legend()
        
        # Cumulative distribution
        sorted_lengths = sorted(lengths)
        cumulative = [i / len(sorted_lengths) * 100 for i in range(len(sorted_lengths))]
        axes[1, 1].plot(sorted_lengths, cumulative)
        axes[1, 1].set_xlabel('Chunk Length (characters)')
        axes[1, 1].set_ylabel('Cumulative Percentage')
        axes[1, 1].set_title('Cumulative Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
        plt.close()
    
    def print_report(self):
        """Print comprehensive analysis report."""
        print("=" * 80)
        print("CHUNK LENGTH ANALYSIS REPORT")
        print("=" * 80)
        
        # Overall statistics
        stats, lengths, word_counts = self.analyze_chunk_lengths()
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"{'─' * 80}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"\nCharacter Length:")
        print(f"  Min:    {stats['character_length']['min']:>8.0f} chars")
        print(f"  Max:    {stats['character_length']['max']:>8.0f} chars")
        print(f"  Mean:   {stats['character_length']['mean']:>8.2f} chars")
        print(f"  Median: {stats['character_length']['median']:>8.0f} chars")
        print(f"  StdDev: {stats['character_length']['stdev']:>8.2f} chars")
        
        print(f"\nWord Count:")
        print(f"  Min:    {stats['word_count']['min']:>8.0f} words")
        print(f"  Max:    {stats['word_count']['max']:>8.0f} words")
        print(f"  Mean:   {stats['word_count']['mean']:>8.2f} words")
        print(f"  Median: {stats['word_count']['median']:>8.0f} words")
        print(f"  StdDev: {stats['word_count']['stdev']:>8.2f} words")
        
        # Coefficient of Variation (CV) - lower is more uniform
        cv = (stats['character_length']['stdev'] / stats['character_length']['mean']) * 100
        print(f"\n📈 Coefficient of Variation: {cv:.2f}%")
        if cv < 10:
            print("   ✅ Very uniform chunks")
        elif cv < 20:
            print("   ✓ Reasonably uniform chunks")
        elif cv < 30:
            print("   ⚠ Moderate variation in chunk sizes")
        else:
            print("   ❌ High variation in chunk sizes")
        
        # Source-wise analysis
        print(f"\n📁 BY SOURCE FILE")
        print(f"{'─' * 80}")
        source_stats = self.analyze_by_source()
        for source, stats in source_stats.items():
            print(f"\n{source}:")
            print(f"  Chunks: {stats['count']}")
            print(f"  Avg Length: {stats['mean']:.0f} chars (±{stats['stdev']:.0f})")
            print(f"  Range: {stats['min']}-{stats['max']} chars")
        
        # Page analysis
        print(f"\n📄 BY PAGE")
        print(f"{'─' * 80}")
        by_page, page_stats = self.analyze_by_page()
        print(f"Average chunks per page: {page_stats['avg_chunks_per_page']:.2f}")
        print(f"Range: {page_stats['min_chunks_per_page']}-{page_stats['max_chunks_per_page']} chunks/page")
        
        # Uniformity check
        print(f"\n⚠️  OUTLIER DETECTION (±20% from mean)")
        print(f"{'─' * 80}")
        outliers, mean_length, tolerance = self.check_uniformity(tolerance_percent=20)
        print(f"Mean Length: {mean_length:.0f} chars")
        print(f"Tolerance: ±{tolerance:.0f} chars ({mean_length-tolerance:.0f} - {mean_length+tolerance:.0f})")
        print(f"Outliers Found: {len(outliers)} ({len(outliers)/len(self.chunks)*100:.1f}%)")
        
        if outliers:
            print(f"\nTop 10 Outliers:")
            for outlier in sorted(outliers, key=lambda x: abs(x['deviation']), reverse=True)[:10]:
                print(f"  • {outlier['id']}: {outlier['length']} chars "
                      f"({outlier['deviation_percent']:+.1f}% from mean)")
        else:
            print("  ✅ All chunks within tolerance!")
        
        print("\n" + "=" * 80)
    
    def export_analysis(self, output_file='chunk_analysis.json'):
        """Export analysis results to JSON."""
        stats, lengths, word_counts = self.analyze_chunk_lengths()
        source_stats = self.analyze_by_source()
        by_page, page_stats = self.analyze_by_page()
        outliers, mean_length, tolerance = self.check_uniformity()
        
        analysis = {
            'overall_statistics': stats,
            'by_source': source_stats,
            'page_statistics': page_stats,
            'uniformity_check': {
                'mean_length': mean_length,
                'tolerance': tolerance,
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers) / len(self.chunks)) * 100,
                'outliers': outliers[:20]  # Top 20 outliers
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Analysis exported to: {output_file}")


# Usage Example
if __name__ == "__main__":
    # Replace with your chunks file path
    chunks_file = "../outputs/chunks.json"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Please provide the correct path to your chunks.json file")
    else:
        analyzer = ChunkAnalyzer(chunks_file)
        
        # Print comprehensive report
        analyzer.print_report()
        
        # Create visualizations
        analyzer.visualize_distribution()
        
        # Export analysis
        analyzer.export_analysis()