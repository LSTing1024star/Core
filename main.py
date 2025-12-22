import os
import sys
current_file_path=os.path.abspath(__file__)
parent_path=os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_path)
import src.config as config

from src.recommend import NodeLevelRecommender
from utils.eval_utils import evaluate_recommendation_comprehensive,save_recommendation_detail

def task2_recommendation():
    print("\n========= 开始执行任务2: 榜样推荐 ===========")
    try:
        recommender = NodeLevelRecommender(
            train_path=config.TRAIN_PATH,
            test_path=config.TEST_PATH
        )
        result_df=recommender.generate_recommendations()
    except Exception as e:
        print(f"运行失败：{str(e)}")

    train_feat=recommender.get_train_node_feat()
    test_feat=recommender.get_test_node_feat()
    core_feats=recommender.get_core_feats()

    evaluate_recommendation_comprehensive(
        result_df,test_feat,train_feat,core_feats,
        k=5,sample_rate=0.2
    )

    save_recommendation_detail(result_df,train_feat,save_path=f"{config.RESULTHOME}hi_recommend2.csv")
    print("========== 任务2执行完毕 ==========\n")

task2_recommendation()