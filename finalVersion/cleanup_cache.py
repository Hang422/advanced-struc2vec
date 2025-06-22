#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存清理工具
清理实验过程中产生的临时文件和缓存
"""
import os
import shutil
import argparse
from pathlib import Path

def get_cache_size(directory):
    """计算目录大小"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except:
        pass
    return total_size

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def clean_directory(directory, description, dry_run=False):
    """清理指定目录"""
    if not directory.exists():
        print(f"   📁 {description}: 目录不存在")
        return 0
    
    # 统计文件
    files = list(directory.glob('*'))
    if not files:
        print(f"   📁 {description}: 已经是空的")
        return 0
    
    total_size = get_cache_size(directory)
    
    if dry_run:
        print(f"   📁 {description}: {len(files)} 个文件, {format_size(total_size)} (预览模式)")
        for file in files[:5]:  # 只显示前5个
            print(f"      - {file.name}")
        if len(files) > 5:
            print(f"      - ... 还有 {len(files) - 5} 个文件")
        return total_size
    else:
        print(f"   🧹 清理 {description}: {len(files)} 个文件, {format_size(total_size)}")
        try:
            shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
            return total_size
        except Exception as e:
            print(f"      ❌ 清理失败: {e}")
            return 0

def cleanup_cache(dry_run=False, selective=False):
    """清理缓存文件"""
    project_root = Path(__file__).parent
    
    # 缓存目录定义
    cache_dirs = {
        'temp': {
            'path': project_root / 'temp',
            'description': 'ORCA Graphlet 缓存',
            'risk': '低风险 - 重新运行时会自动重新计算'
        },
        'distances': {
            'path': project_root / 'output' / 'distances', 
            'description': '距离矩阵缓存',
            'risk': '中风险 - 需要重新计算距离矩阵'
        },
        'temp_struc2vec': {
            'path': project_root / 'temp_struc2vec',
            'description': 'Struc2Vec 临时文件',
            'risk': '低风险 - Struc2Vec 内部缓存'
        },
        'embeddings': {
            'path': project_root / 'output' / 'embeddings',
            'description': '嵌入向量缓存', 
            'risk': '高风险 - 需要重新训练模型'
        },
        'results': {
            'path': project_root / 'output' / 'results',
            'description': '实验结果缓存',
            'risk': '高风险 - 实验结果会丢失'
        }
    }
    
    print("🧹 缓存清理工具")
    print("=" * 60)
    
    if dry_run:
        print("📋 预览模式 - 不会删除任何文件")
    else:
        print("⚠️  清理模式 - 将删除选中的缓存文件")
    
    total_cleaned = 0
    
    for key, info in cache_dirs.items():
        if selective:
            # 选择性清理
            current_size = get_cache_size(info['path'])
            if current_size == 0:
                continue
                
            print(f"\\n📁 {info['description']}")
            print(f"   路径: {info['path']}")
            print(f"   大小: {format_size(current_size)}")
            print(f"   风险: {info['risk']}")
            
            if not dry_run:
                response = input(f"   清理这个目录吗? (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    cleaned = clean_directory(info['path'], info['description'], dry_run=False)
                    total_cleaned += cleaned
                else:
                    print(f"   ⏭️  跳过 {info['description']}")
            else:
                cleaned = clean_directory(info['path'], info['description'], dry_run=True)
                total_cleaned += cleaned
        else:
            # 批量清理
            cleaned = clean_directory(info['path'], info['description'], dry_run)
            total_cleaned += cleaned
    
    print("\\n" + "=" * 60)
    if dry_run:
        print(f"📊 预览结果: 总共 {format_size(total_cleaned)} 可以清理")
        print("💡 运行 'python cleanup_cache.py --clean' 进行实际清理")
    else:
        print(f"✅ 清理完成: 释放了 {format_size(total_cleaned)} 空间")

def main():
    parser = argparse.ArgumentParser(description='缓存清理工具')
    parser.add_argument('--clean', action='store_true', 
                       help='实际清理文件 (默认只预览)')
    parser.add_argument('--selective', action='store_true',
                       help='选择性清理 (逐个确认)')
    parser.add_argument('--all', action='store_true',
                       help='清理所有缓存 (危险!)')
    
    args = parser.parse_args()
    
    if args.all and not args.clean:
        print("❌ --all 选项必须与 --clean 一起使用")
        return
    
    if args.all:
        print("⚠️  警告: 将清理所有缓存文件!")
        response = input("确定要继续吗? 输入 'DELETE ALL' 确认: ")
        if response != 'DELETE ALL':
            print("取消清理")
            return
    
    dry_run = not args.clean
    selective = args.selective and not args.all
    
    cleanup_cache(dry_run=dry_run, selective=selective)

if __name__ == "__main__":
    main()