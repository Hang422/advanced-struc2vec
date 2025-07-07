# 在你的脚本开头添加：
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.utils import struc2vec_patch  # 自动应用补丁

# 然后正常使用 Struc2Vec
from GraphEmbedding.ge.models.struc2vec import Struc2Vec
