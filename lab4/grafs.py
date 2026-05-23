"""
Скрипт для визуализации результатов бенчмарка CUDA matrix multiplication.
Читает results.txt и строит графики:
  1. GFLOPS vs N (размер матрицы) для разных конфигураций блока
  2. Время выполнения (мс) vs N
  3. Сравнение конфигураций (bar chart) для каждого N
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = (10, 6)

BLOCK_COLORS = {
    '8x8': '#FF6B6B',
    '16x16': '#4ECDC4',
    '32x32': '#95E1D3'
}

def parse_results(filename: str) -> dict:
    """
    Парсит файл результатов и возвращает структурированные данные.
    Returns: {block_size: {N: {'time': ..., 'gflops': ...}}}
    """
    data = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Using GPU') or line.startswith('N,') or \
               line.startswith('[GPU]') or line.startswith('[DONE]'):
                continue
            
            match = re.match(r'^(\d+),(\d+x\d+),([\d.]+),([\d.]+),(.+)$', line)
            if match:
                n = int(match.group(1))
                block = match.group(2)
                time_sec = float(match.group(3))
                gflops = float(match.group(4))
                
                if block not in data:
                    data[block] = {}
                data[block][n] = {'time_ms': time_sec * 1000, 'gflops': gflops}
    
    return data


def plot_gflops_vs_n(data: dict, output: str = 'gflops_vs_n.png'):
    """График: Производительность (GFLOPS) в зависимости от размера матрицы."""
    plt.figure()
    
    for block, results in data.items():
        sizes = sorted(results.keys())
        gflops = [results[n]['gflops'] for n in sizes]
        label = f'Блок {block}' + ('' if block == '16x16' else '')
        plt.plot(sizes, gflops, 'o-', label=label, 
                color=BLOCK_COLORS.get(block, '#333'), linewidth=2, markersize=6)
    
    plt.xlabel('Размер матрицы N × N', fontsize=12)
    plt.ylabel('Производительность (GFLOPS)', fontsize=12)
    plt.title('Производительность умножения матриц на GTX 1650 Super\n(наивное CUDA-ядро, double)', fontsize=14)
    plt.legend(frameon=True, fancybox=True)
    plt.grid(alpha=0.3)
    
    plt.axhline(y=137, color='gray', linestyle='--', alpha=0.5, label='Пик FP64 (теор.)')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Сохранён: {output}")
    plt.close()


def plot_time_vs_n(data: dict, output: str = 'time_vs_n.png'):
    """График: Время выполнения (мс) в зависимости от размера матрицы."""
    plt.figure()
    
    for block, results in data.items():
        sizes = sorted(results.keys())
        times = [results[n]['time_ms'] for n in sizes]
        label = f'Блок {block}' + ('' if block == '16x16' else '')
        plt.plot(sizes, times, 's--', label=label, 
                color=BLOCK_COLORS.get(block, '#333'), linewidth=2, markersize=6)
    
    plt.xlabel('Размер матрицы N × N', fontsize=12)
    plt.ylabel('Время выполнения (мс)', fontsize=12)
    plt.title('Время выполнения ядра CUDA в зависимости от размера задачи', fontsize=14)
    plt.legend(frameon=True, fancybox=True)
    plt.grid(alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Сохранён: {output}")
    plt.close()


def plot_comparison_bar(data: dict, output: str = 'comparison_bars.png'):
    """Сравнительная гистограмма: лучшая конфигурация для каждого N."""
    sizes = sorted(next(iter(data.values())).keys())
    x = np.arange(len(sizes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, (block, results) in enumerate(data.items()):
        gflops = [results[n]['gflops'] for n in sizes]
        label = f'Блок {block}' + ('оптимальная' if block == '16x16' else '')
        ax.bar(x + i*width - width, gflops, width, label=label, 
              color=BLOCK_COLORS.get(block, '#333'), edgecolor='white', linewidth=1.2)
    
    ax.set_xlabel('Размер матрицы N × N', fontsize=12)
    ax.set_ylabel('Производительность (GFLOPS)', fontsize=12)
    ax.set_title('Сравнение конфигураций блоков для каждого размера матрицы', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(frameon=True, fancybox=True)
    ax.grid(axis='y', alpha=0.3)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Сохранён: {output}")
    plt.close()


def plot_efficiency(data: dict, output: str = 'efficiency.png'):
    """График: Эффективность использования пика (в % от 137 GFLOPS)."""
    PEAK_FP64 = 137.0
    
    plt.figure(figsize=(10, 6))
    
    for block, results in data.items():
        sizes = sorted(results.keys())
        efficiency = [results[n]['gflops'] / PEAK_FP64 * 100 for n in sizes]
        label = f'Блок {block}' + ('' if block == '16x16' else '')
        plt.plot(sizes, efficiency, 'd-', label=label, 
                color=BLOCK_COLORS.get(block, '#333'), linewidth=2, markersize=6)
    
    plt.xlabel('Размер матрицы N × N', fontsize=12)
    plt.ylabel('Эффективность (% от пика FP64)', fontsize=12)
    plt.title('Эффективность использования вычислительных ресурсов GPU', fontsize=14)
    plt.axhline(y=100, color='red', linestyle=':', alpha=0.3, label='100% пика')
    plt.legend(frameon=True, fancybox=True)
    plt.grid(alpha=0.3)
    plt.ylim(0, 50)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Сохранён: {output}")
    plt.close()


def print_summary(data: dict):
    """Выводит текстовую сводку в консоль."""
    print("\n" + "="*70)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("="*70)
    
    for block in sorted(data.keys()):
        results = data[block]
        sizes = sorted(results.keys())
        best_n = max(sizes, key=lambda n: results[n]['gflops'])
        best_gflops = results[best_n]['gflops']
        
        print(f"\n🔹 Конфигурация блока {block}:")
        print(f"   • Лучший результат: {best_gflops:.2f} GFLOPS при N={best_n}")
        print(f"   • Диапазон: {min(r['gflops'] for r in results.values()):.2f} – {max(r['gflops'] for r in results.values()):.2f} GFLOPS")
    
    print(f"\nОптимальная конфигурация: 16×16")
    best_16x16 = data['16x16']
    for n in sorted(best_16x16.keys()):
        gain_8x8 = (best_16x16[n]['gflops'] / data['8x8'][n]['gflops'] - 1) * 100
        gain_32x32 = (best_16x16[n]['gflops'] / data['32x32'][n]['gflops'] - 1) * 100
        print(f"   N={n:4d}: +{gain_8x8:5.1f}% vs 8×8,  +{gain_32x32:5.1f}% vs 32×32")
    
    print("\n" + "="*70)


def main():
    results_file = 'results.txt'
    
    if not Path(results_file).exists():
        print(f"Файл '{results_file}' не найден!")
        print(" Поместите файл с результатами в ту же папку, что и этот скрипт.")
        return
    
    print(f"📊 Чтение данных из {results_file}...")
    data = parse_results(results_file)
    
    if not data:
        print(" Не удалось распарсить данные. Проверьте формат файла.")
        return
    
    print(f"✓ Загружено {len(data)} конфигураций: {list(data.keys())}")
    
    print("\n Построение графиков...")
    plot_gflops_vs_n(data)
    plot_time_vs_n(data)
    plot_comparison_bar(data)
    plot_efficiency(data)
    
    print_summary(data)
    
    print("\n Готово! Графики сохранены в текущей папке.")
    print(" Откройте файлы .png для просмотра.")


if __name__ == '__main__':
    main()