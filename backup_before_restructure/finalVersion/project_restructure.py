#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目重构工具
清理项目结构，删除重复文件，整理为最终版本
"""
import os
import shutil
import subprocess
from pathlib import Path

class ProjectRestructure:
    """项目重构器"""
    
    def __init__(self):
        self.finalVersion_dir = Path(__file__).parent
        self.root_dir = self.finalVersion_dir.parent
        self.backup_dir = self.root_dir / "backup_before_restructure"
        
    def analyze_project_structure(self):
        """分析当前项目结构"""
        print("🔍 分析当前项目结构...")
        
        # 分析根目录文件
        root_files = list(self.root_dir.glob("*"))
        print(f"\n📁 根目录文件/目录数量: {len(root_files)}")
        
        # 分类文件
        categories = {
            "数据目录": [],
            "算法目录": [],
            "依赖项目": [],
            "配置文件": [],
            "临时文件": [],
            "脚本文件": [],
            "文档文件": [],
            "其他文件": []
        }
        
        for item in root_files:
            if item.name.startswith('.'):
                continue
            elif item.name in ['data', 'algorithms']:
                categories["数据目录" if item.name == 'data' else "算法目录"].append(item)
            elif item.name in ['GraphEmbedding', 'orca']:
                categories["依赖项目"].append(item)
            elif item.name in ['requirements.txt', 'README.md', 'IMPLEMENTATION_SUMMARY.md']:
                categories["配置文件"].append(item)
            elif item.name in ['temp_struc2vec', 'output']:
                categories["临时文件"].append(item)
            elif item.name.endswith('.py'):
                categories["脚本文件"].append(item)
            elif item.name.endswith('.md'):
                categories["文档文件"].append(item)
            else:
                categories["其他文件"].append(item)
        
        for category, items in categories.items():
            if items:
                print(f"\n{category}:")
                for item in items:
                    size = self._get_size(item)
                    print(f"  - {item.name} ({size})")
        
        return categories
    
    def _get_size(self, path):
        """获取文件/目录大小"""
        if path.is_file():
            size = path.stat().st_size
            if size < 1024:
                return f"{size} B"
            elif size < 1024**2:
                return f"{size/1024:.1f} KB"
            elif size < 1024**3:
                return f"{size/1024**2:.1f} MB"
            else:
                return f"{size/1024**3:.1f} GB"
        elif path.is_dir():
            try:
                result = subprocess.run(['du', '-sh', str(path)], 
                                      capture_output=True, text=True)
                return result.stdout.split()[0]
            except:
                return "目录"
        return "未知"
    
    def create_backup(self):
        """创建备份"""
        print(f"\n💾 创建备份到: {self.backup_dir}")
        
        if self.backup_dir.exists():
            print(f"备份目录已存在，自动覆盖...")
            shutil.rmtree(self.backup_dir)
        
        # 备份重要文件
        self.backup_dir.mkdir()
        
        important_items = [
            'finalVersion',
            'GraphEmbedding', 
            'orca',
            'data',
            'algorithms',
            'requirements.txt',
            'README.md'
        ]
        
        for item_name in important_items:
            item_path = self.root_dir / item_name
            if item_path.exists():
                if item_path.is_dir():
                    shutil.copytree(item_path, self.backup_dir / item_name)
                else:
                    shutil.copy2(item_path, self.backup_dir / item_name)
                print(f"  ✅ {item_name}")
        
        print("✅ 备份完成")
        return True
    
    def plan_restructure(self):
        """制定重构计划"""
        print("\n📋 制定重构计划...")
        
        plan = {
            "保留并整理": {
                "finalVersion/": "主要项目目录",
                "libs/GraphEmbedding/": "图嵌入库(作为模块)",
                "libs/orca/": "ORCA工具(作为模块)", 
                "data/": "数据集",
                "README.md": "项目说明",
                "requirements.txt": "依赖列表"
            },
            "删除重复文件": {
                "根目录/algorithms/": "与finalVersion重复",
                "根目录/data/": "与finalVersion重复",
                "根目录/*.py": "散乱的脚本文件",
                "根目录/temp_struc2vec/": "临时文件",
                "根目录/output/": "旧的输出文件",
                "根目录/evaluation/": "与finalVersion功能重复"
            },
            "移动到libs": {
                "GraphEmbedding/": "移动到libs/GraphEmbedding/",
                "orca/": "移动到libs/orca/"
            },
            "finalVersion结构优化": {
                "run_advanced_fusion_comparison.py": "主要入口脚本",
                "src/": "核心算法实现",
                "data/": "数据集",
                "config/": "配置文件",
                "scripts/": "辅助脚本"
            }
        }
        
        for action, items in plan.items():
            print(f"\n{action}:")
            for item, desc in items.items():
                print(f"  - {item}: {desc}")
        
        return plan
    
    def execute_restructure(self, plan):
        """执行重构"""
        print("\n🚀 开始执行重构...")
        
        # 1. 创建新的结构
        print("\n1️⃣ 创建新的目录结构...")
        libs_dir = self.root_dir / "libs"
        libs_dir.mkdir(exist_ok=True)
        
        # 2. 移动GraphEmbedding和orca到libs
        print("\n2️⃣ 移动依赖库到libs/...")
        
        graphembedding_src = self.root_dir / "GraphEmbedding"
        graphembedding_dst = libs_dir / "GraphEmbedding"
        if graphembedding_src.exists() and not graphembedding_dst.exists():
            shutil.move(str(graphembedding_src), str(graphembedding_dst))
            print(f"  ✅ 移动 GraphEmbedding -> libs/GraphEmbedding")
        
        orca_src = self.root_dir / "orca"
        orca_dst = libs_dir / "orca"
        if orca_src.exists() and not orca_dst.exists():
            shutil.move(str(orca_src), str(orca_dst))
            print(f"  ✅ 移动 orca -> libs/orca")
        
        # 3. 删除重复和无用文件
        print("\n3️⃣ 删除重复和无用文件...")
        
        items_to_delete = [
            "algorithms",
            "advanced_fusion.py",
            "evaluate_fusion.py", 
            "evaluation",
            "generate_graphlet_distances.py",
            "generate_simple_distance.py",
            "optimize_fusion.py",
            "output",
            "run_comparison.py",
            "simple_evaluation.py",
            "simple_fusion_evaluation.py",
            "temp_struc2vec",
            "test_advanced_fusion.py",
            "test_existing_fusion.py",
            "IMPLEMENTATION_SUMMARY.md"
        ]
        
        for item_name in items_to_delete:
            item_path = self.root_dir / item_name
            if item_path.exists():
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                else:
                    item_path.unlink()
                print(f"  🗑️  删除 {item_name}")
        
        # 4. 更新finalVersion中的导入路径
        print("\n4️⃣ 更新导入路径...")
        self._update_import_paths()
        
        # 5. 创建新的项目根目录README
        print("\n5️⃣ 创建项目文档...")
        self._create_project_readme()
        
        print("\n✅ 重构完成!")
    
    def _update_import_paths(self):
        """更新finalVersion中的导入路径"""
        
        # 需要更新的文件
        files_to_update = [
            self.finalVersion_dir / "run_advanced_fusion_comparison.py",
            self.finalVersion_dir / "run_simple_comparison.py", 
            self.finalVersion_dir / "run_safe_comparison.py",
            self.finalVersion_dir / "src" / "algorithms" / "advanced_fusion_methods.py"
        ]
        
        # 路径映射
        path_mappings = {
            "from GraphEmbedding.ge": "from libs.GraphEmbedding.ge",
            "from algorithms.traditional.struc2vec": "from libs.GraphEmbedding.ge.models.struc2vec", 
            "from algorithms.graphlet_based.compute_edges_improved": "from algorithms.graphlet_based.compute_edges_improved"
        }
        
        for file_path in files_to_update:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    modified = False
                    for old_import, new_import in path_mappings.items():
                        if old_import in content:
                            content = content.replace(old_import, new_import)
                            modified = True
                    
                    if modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"    ✅ 更新 {file_path.name}")
                        
                except Exception as e:
                    print(f"    ❌ 更新失败 {file_path.name}: {e}")
    
    def _create_project_readme(self):
        """创建新的项目README"""
        readme_content = '''# Advanced Graphlet-Enhanced Struc2Vec

## 项目简介

这是一个增强版的Struc2Vec项目，集成了graphlet特征和多种先进的特征融合方法。

## 核心功能

- **原始Struc2Vec算法**: 基于度序列的结构相似性
- **Graphlet增强版本**: 使用ORCA工具计算graphlet特征
- **5种高级融合方法**: 
  - 多头注意力融合
  - 层次化金字塔融合
  - 谱感知融合
  - 社区感知融合
  - 集成融合

## 项目结构

```
graphlet_struc2vec/
├── finalVersion/           # 主要项目目录
│   ├── run_advanced_fusion_comparison.py  # 🚀 主要入口脚本
│   ├── src/               # 核心算法实现
│   ├── data/              # 数据集
│   ├── config/            # 配置文件
│   └── scripts/           # 辅助脚本
├── libs/                  # 依赖库
│   ├── GraphEmbedding/    # 图嵌入库
│   └── orca/              # ORCA graphlet工具
├── data/                  # 数据集
└── README.md             # 项目说明
```

## 快速开始

```bash
# 基础比较
cd finalVersion
python run_advanced_fusion_comparison.py --dataset europe-airports

# 高级融合方法比较
python run_advanced_fusion_comparison.py --dataset europe-airports --fusion-methods attention,spectral,community,ensemble

# 自定义参数
python run_advanced_fusion_comparison.py --dataset wiki --fusion-methods attention --num-walks 10 --walk-length 50
```

## 支持的数据集

- **europe-airports**: 欧洲机场网络
- **wiki**: Wikipedia网络
- **lastfm**: Last.fm音乐网络

## 依赖

- Python 3.7+
- NetworkX
- NumPy
- scikit-learn
- ORCA (已包含)

## 算法说明

### 融合方法对比

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| 注意力融合 | 自适应权重学习 | 复杂异构图 |
| 金字塔融合 | 多尺度信息处理 | 层次化结构 |
| 谱感知融合 | 基于图谱特性 | 社区结构明显 |
| 社区感知融合 | 社区内外差异化 | 社交网络 |
| 集成融合 | 多方法组合 | 追求最佳性能 |

## 引用

如果这个项目对您的研究有帮助，请考虑引用：

```bibtex
@misc{advanced_graphlet_struc2vec,
  title={Advanced Graphlet-Enhanced Struc2Vec with Multiple Fusion Strategies},
  year={2024}
}
```

## 许可证

MIT License
'''
        
        readme_path = self.root_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print("  ✅ 创建项目README.md")
    
    def verify_restructure(self):
        """验证重构结果"""
        print("\n🔍 验证重构结果...")
        
        # 检查新结构
        expected_structure = {
            "finalVersion/": "主项目目录",
            "libs/GraphEmbedding/": "图嵌入库",
            "libs/orca/": "ORCA工具",
            "data/": "数据集",
            "README.md": "项目说明"
        }
        
        all_good = True
        for path, desc in expected_structure.items():
            full_path = self.root_dir / path
            if full_path.exists():
                print(f"  ✅ {path}: {desc}")
            else:
                print(f"  ❌ {path}: 缺失!")
                all_good = False
        
        # 测试主脚本
        print(f"\n🧪 测试主脚本...")
        main_script = self.finalVersion_dir / "run_advanced_fusion_comparison.py"
        if main_script.exists():
            print(f"  ✅ 主脚本存在: {main_script}")
            print(f"  💡 运行测试: cd finalVersion && python run_advanced_fusion_comparison.py --help")
        else:
            print(f"  ❌ 主脚本缺失!")
            all_good = False
        
        if all_good:
            print(f"\n🎉 重构验证成功!")
        else:
            print(f"\n⚠️  重构可能有问题，请检查")
        
        return all_good

def main():
    """主函数"""
    restructure = ProjectRestructure()
    
    print("🚀 项目重构工具")
    print("=" * 60)
    
    # 分析当前结构
    categories = restructure.analyze_project_structure()
    
    # 制定计划
    plan = restructure.plan_restructure()
    
    # 确认执行
    print(f"\n❓ 执行重构计划")
    print(f"   这将删除根目录下的重复文件，整理项目结构")
    print(f"   重要文件会先备份到 backup_before_restructure/")
    
    print(f"\n✅ 自动执行重构...")
    
    # 创建备份
    if not restructure.create_backup():
        return
    
    # 执行重构
    restructure.execute_restructure(plan)
    
    # 验证结果
    restructure.verify_restructure()
    
    print(f"\n🎯 重构完成!")
    print(f"   主要入口: finalVersion/run_advanced_fusion_comparison.py")
    print(f"   备份位置: backup_before_restructure/")

if __name__ == "__main__":
    main()