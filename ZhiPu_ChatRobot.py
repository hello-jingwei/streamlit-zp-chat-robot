from langchain_community.chat_models import ChatZhipuAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
import os
import streamlit as st # 快速页面构建


# 设置API Key（建议使用环境变量）
os.environ['ZHIPUAI_API_KEY'] = '44287bbb579a43d1a7db2371f65ef1d3.xIh4V1zTF4unRaa8'

st.set_page_config(page_title='智普智能聊天机器人')
st.title('智普智能聊天机器人')

with st.sidebar:
    st.header("说明")

# 初始化智谱AI聊天模型
chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.7,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的AI助手"),
    ("human", "{input}")
])

# 创建链
chain = prompt | chat

print("智谱AI聊天机器人已启动！输入'退出'或'quit'结束对话。")

while True:
    user_input = input("\n用户: ")
    if user_input.lower() in ["退出", "quit"]:
        print("再见！")
        break

    try:
        # 调用链
        response = chain.invoke({"input": user_input})
        print()  # 换行
    except Exception as e:
        print(f"\n出错了: {e}")