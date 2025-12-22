import os
import sys
current_file_path=os.path.abspath(__file__)
parent_path=os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_path)
from src.config import(
    RAW_DATA_PATH,TRAIN_PATH,TEST_PATH,NEW_STUDENT_DATA_PATH,
    TEST_TERMS,NEW_STUDENT_TERM,GPA_EMPTY_MARK,
    DATAHOME,RESULTHOME,ENCODING
)
from utils.data_utils import load_and_preprocess_data,build_node_level_samples
from utils.llm_utils import predict_node_level, predict_new_students
from utils.eval_utils import evaluate_prediction,evaluate_new_students_prediction,save_metrics_summary

def main():
    print("\n===== 加载数据集 =====")
    df_train=load_and_preprocess_data(TRAIN_PATH)
    df_test=load_and_preprocess_data(TEST_PATH)

    print("\n===== 处理Test集 =====")
    test_sample_df=build_node_level_samples(df_test)
    test_result_df=predict_node_level(test_sample_df)
    test_result_df.to_csv(f"{RESULTHOME}hi.csv")
    test_metrics=evaluate_prediction(test_result_df,dataset_name="Test")
    test_result_df.to_csv(f"{RESULTHOME}test_node_prediction.csv",index=False,encoding=ENCODING)
    
    print("\n所有任务完成！")

if __name__=="__main__":
    main()