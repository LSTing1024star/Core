import re
import requests
import json
from openai import openai
from termcolor import colored

import os
import sys
current_file_path=os.path.abspath(__file__)
parent_path=os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_path)
from config import (
    LLM_API_URL, LLM_MAX_LENGTH, LLM_TEMPERATURE, LLM_TOP_P,
    TARGET_COL, NEW_STUDENT_TERM, NEW_STUDENT_PRED_PATH, ENCODING,
    GPA_EMPTY_MARK, LLM_TEMPERATURE, LLM_MAX_NEW_TOKENS
)

client=OpenAI(
    api_key="dummy-key",
    base_url=LLM_API_URL
)

def call_local_llm(prompt,task=1):
    """调用本地大模型（通用接口）"""
    try:
        response=client.chat.completions.create(
            model="qwen8b",
            messages=[
                {"role":"system","content":"所有回答直接输出结果，不要think。"},
                {"role":"user","content":prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_NEW_TOKENS
        )
        result=response.choices[0].message.content
        if task=1:
            result=result[-1]
        print(colored(result,"red"))
        return result.strip()
    except Exception as e:
        print(f"LLM调用失败：{str(e)}，默认预测为0")
        return "0"

def construct_prediction_prompt(feature_desc):
    """常规学生的预测Prompt（适配GPA空值标识）"""
    prompt = f"""
    你是学业预警预测专家，需要根据学生的历史学业特征，预测其当前学期是否会出现学业预警（{TARGET_COL}）。
    预测规则：
    1. 仅输出数字0或1（0=无学业预警，1=有学业预警），无需额外解释。
    2. 基于历史特征的趋势和数值大小进行判断（如GPA持续下降、挂科数增加则更可能预警）。
    3. 若GPA为「{GPA_EMPTY_MARK}」，则忽略GPA，仅根据挂科数/违纪数、奖学金次数判断。
    
    学生历史特征：{feature_desc}
    
    预测结果：
    """
    return prompt.strip()

def construct_new_student_prompt(student_data):
    """纯新入学学生的预测Prompt（适配split.py的数据集结构）"""
    prompt = f"""
    你是学业预警预测专家，需要根据2025年新入学学生的入学特征，预测其当前学期是否会出现学业预警（{TARGET_COL}）。
    预测规则：
    1. 仅输出数字0或1（0=无学业预警，1=有学业预警），无需额外解释。
    2. 新入学学生GPA为「{GPA_EMPTY_MARK}」，仅根据挂科数/违纪数、奖学金次数判断（若无挂科数则默认预测0）。
    
    学生信息：
    - 学号：{student_data['XH']}
    - 入学学期：{student_data['XNXQ']}
    - 挂科/违纪数：{student_data['DCCY']}
    - 奖学金次数：{student_data['JXJ']}
    
    预测结果：
    """
    return prompt.strip()

def predict_node_level(sample_df):
    """常规学生的节点预测"""
    predictions = []
    for idx, row in sample_df.iterrows():
        prompt = construct_prediction_prompt(row["history_feature_desc"])
        llm_output = call_local_llm(prompt)
        pred = re.findall(r"[01]", llm_output)
        pred_label = int(pred[0]) if pred else 0
        predictions.append(pred_label)
        
        if (idx + 1) % 50 == 0:
            print(f"已完成{idx + 1}/{len(sample_df)}个节点的预测")
    
    sample_df["pred_label"] = predictions
    return sample_df

def predict_new_students(new_student_df):
    """纯新入学学生的预测接口（适配split.py的数据集）"""
    if len(new_student_df) == 0:
        print("无纯新入学学生数据，跳过预测")
        return new_student_df
    
    predictions = []
    for idx, row in new_student_df.iterrows():
        prompt = construct_new_student_prompt(row)
        llm_output = call_local_llm(prompt)
        pred = re.findall(r"[01]", llm_output)
        pred_label = int(pred[0]) if pred else 0
        predictions.append(pred_label)
        
        if (idx + 1) % 10 == 0:
            print(f"已完成{idx + 1}/{len(new_student_df)}个新入学学生的预测")
    
    new_student_df["pred_label"] = predictions
    new_student_df.to_csv(NEW_STUDENT_PRED_PATH, index=False, encoding=ENCODING)
    print(f"\n纯新入学学生预测结果已保存至：{NEW_STUDENT_PRED_PATH}")
    
    return new_student_df