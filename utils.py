import pandas as pd
import joblib
from langchain_community.chat_models import ChatZhipuAI
from langchain.prompts import ChatPromptTemplate
import os

def fill_info(info, hypertension_level=None, BMI=None, non_HDL=None, crp=None, age=None,
       gender=None, ethnicity=None, education=None, INDFMPIR=None, diabetes=None,
       diet_score=None, MCQ300A=None, MCQ300C=None, physical_level=None,
       sedentary_minutes=None, sleep_level=None, drinks=None, smoke=None, hbac1=None, MCQ220=None, MCQ160C=None, MCQ160F=None):
    
    if hypertension_level:
        if hypertension_level == 'Yes':
            hypertension_level = 1
        else:
            hypertension_level = 0
        info.loc[0, 'hypertension_level'] = hypertension_level

    if BMI:
        info.loc[0, 'BMXBMI'] = BMI
    if non_HDL:
        info.loc[0, 'non_HDL'] = non_HDL
    if crp:
        info.loc[0, 'LBXCRP'] = crp
    if age:
        info.loc[0, 'RIDAGEYR'] = age
    if gender:
        if gender == 'Female':
            gender1 = 0
        else:
            gender1 = 1
        info.loc[0, 'RIAGENDR'] = gender1
    if ethnicity:
        lst = ['Mexican American', 'Other Hispanic', 'Non-Hispanic White', 'Non-Hispanic Black', 'Other Race - Including Multi-Racial']
        ethnicity1 = lst.index(ethnicity)
        info.loc[0, 'RIDRETH1'] = ethnicity1
    if education:
        lst = ['Less Than 9th Grade', '9-11th Grade', 'High School Grad/GED or Equivalent', 'Some College or AA degree', 'College Graduate or above']
        education1 = lst.index(education)
        info.loc[0, 'DMDEDUC2'] = education1
    if INDFMPIR:
        info.loc[0, 'INDFMPIR'] = INDFMPIR
    if diabetes:
        lst = ['No diabetes','Diabetes', 'Prediabetes']
        diabetes1 = lst.index(diabetes)
        info.loc[0, 'diabetes_level'] = diabetes1
    if diet_score:
        info.loc[0, 'diet_score'] = diet_score
    if MCQ300A:
        if MCQ300A == 'Yes':
            MCQ300A1=1
        else: 
            MCQ300A1=0
        info.loc[0, 'MCQ300A'] = MCQ300A1
    if MCQ300C:
        if MCQ300C == 'Yes':
            MCQ300C1=1
        else: 
            MCQ300C1=0
        info.loc[0, 'MCQ300C'] = MCQ300C1
    if physical_level:
        lst = ['Inactive', 'insufficient active', 'active', 'highly active']
        physical_level1 = lst.index(physical_level)
        info.loc[0, 'physical_level'] = physical_level1
    if sedentary_minutes:
        info.loc[0, 'Sedentary_minutes'] = sedentary_minutes
    if sleep_level:
        lst = ['Poor', 'Intermediate', 'Optimal']
        sleep_level1 = lst.index(sleep_level)
        info.loc[0, 'sleep_level'] = sleep_level1
    if drinks:
        info.loc[0, 'drinks'] = drinks
    if smoke:
        lst = ['No smoking', 'Smoking before', 'Smoking']
        smoke1 = lst.index(smoke)
        info.loc[0, 'smoking_status'] = smoke1
    if hbac1:
        info.loc[0, 'LBXGH'] = hbac1
    if MCQ220:
        lst = ['Yes', 'No']
        MCQ2201 = lst.index(MCQ220)
        info.loc[0, 'MCQ220'] = MCQ2201
    if MCQ160C:
        lst = ['Yes', 'No']
        MCQ160C1 = lst.index(MCQ160C)
        info.loc[0, 'MCQ160C'] = MCQ160C1
    if MCQ160F:
        lst = ['Yes', 'No']
        MCQ160F1 = lst.index(MCQ160F)
        info.loc[0, 'MCQ160F'] = MCQ160F1   

    return info


def standarscaler(df):
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)
    StandardScaler = joblib.load('StandardScaler-allcause.pkl')
    df_norm = StandardScaler.transform(df)
    df_norm = pd.DataFrame(data=df_norm, columns=StandardScaler.get_feature_names_out())
    print(StandardScaler.get_feature_names_out())
    return df_norm

def ai(allcause_mortality_risk, card_mortality_risk, allcause_shap, card_shap):
    import os
    api_key = os.getenv('ZHIPUAI_API_KEY')
    model = ChatZhipuAI(model='glm-4-flash', api_key=api_key, temperature=1)

    prompt_template = ChatPromptTemplate([
        ('system',"""你是一个机器学习和医学专家，现在我有一个基于生活方式的10年全因死亡和心血管疾病死亡的预测模型，
         我会将该个体的预测死亡风险以及每个变量对应的shap值给你，请你对其进行分析，提出针对于该个体的合理化建议，尤其是对于生活方式方面的。
         回答的内容用markdown格式书写，尽量美观。 回答使用英文。
         
         --------
         回答的格式为，
         标题
         一、概述
         二、评估结果 （这部分简要描述即可）
         三、 建议 （多聚焦于生活方式）

         --------

         其特征分别为：
         1.是否患有高血压：1.有， 2.无', 
         2.BMI值；
         3.非高密度脂蛋白值；
         4.C反应蛋白值；
         5.年龄；
         6.性别：1.男， 0女；
         7.种族：分别为'Mexican American', 'Other Hispanic', 'Non-Hispanic White', 'Non-Hispanic Black', 'Other Race - Including Multi-Racial'；
         8.受教育水平：['Less Than 9th Grade', '9-11th Grade', 'High School Grad/GED or Equivalent', 'Some College or AA degree', 'College Graduate or above']；
         9.收入与平均线的比值；
         10.是否患有糖尿病；分别为'No diabetes','Diabetes', 'Prediabetes'；
         11.饮食评分，根据Healthy Eating Index (HEI) 评价得到的饮食得分；
         12.近亲是否有心脏病：1：有，2：没有；
         13.近亲是否有糖尿病：1：有，2：没有；
         14。体力活动水平：分别是['Inactive', 'insufficient active', 'active', 'highly active']
         15.静坐时间；
         16.睡眠质量：分别是['Poor', 'Intermediate', 'Optimal']；
         17.饮酒量：g/天；
         18.吸烟史：分别是['No smoking', 'Smoking before', 'Smoking']；
         19.体力活动等效运动时间:
         20.糖化血红蛋白；
         21.是否有癌症：1：有，2：没有；
         22.是否有冠心病：1：有，2：没有；
         23.是否有中风：1：有，2：没有；
         """),

        ('user',"""
        该个体的10年全因死亡风险为{allcause_mortality_risk}, 10年心血管疾病死亡风险为{card_mortality_risk},在预测全因死亡方面，其特征所对应的shap值为{allcause_shap},
         在预测心血管疾病死亡方面，其特征对应的shap值为{card_shap}。
""")
    ])

    chain = prompt_template | model
    response = chain.invoke({
        'allcause_mortality_risk':allcause_mortality_risk,
        'card_mortality_risk':card_mortality_risk,
        'allcause_shap': allcause_shap,
        'card_shap': card_shap,
    })

    return response.content


