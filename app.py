import streamlit as st
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
import os
from dotenv import load_dotenv
import asyncio

# 加载环境变量
load_dotenv()


# 自定义回调处理器，用于处理流式输出
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """每次收到新token时调用"""
        self.text += token
        self.container.markdown(self.text + "▌")  # 添加光标效果

    def on_llm_end(self, response, **kwargs) -> None:
        """LLM输出结束时调用"""
        self.container.markdown(self.text)  # 移除光标，显示完整文本


# 页面配置
st.set_page_config(
    page_title="AI聊天机器人",
    page_icon="🤖",
    layout="centered"
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []


def initialize_chat_model(streaming_callback=None):
    """初始化聊天模型"""
    try:
        api_key = os.getenv('ZHIPUAI_API_KEY') or st.secrets.get("ZHIPUAI_API_KEY", "")

        if not api_key:
            st.error("⚠️ 请先设置ZHIPUAI_API_KEY环境变量或在Secrets中配置")
            return None

        # 创建回调管理器
        callback_manager = None
        if streaming_callback:
            callback_manager = CallbackManager([streaming_callback])

        chat = ChatZhipuAI(
            model="glm-4",
            api_key=api_key,
            temperature=0.7,
            top_p=0.9,
            streaming=True,  # 启用流式输出
            callback_manager=callback_manager,  # 使用callback_manager而不是callbacks
        )
        return chat
    except Exception as e:
        st.error(f"初始化模型失败: {str(e)}")
        return None


def get_streaming_response(user_input, message_placeholder):
    """获取流式响应 - 修正版本"""
    try:
        # 创建回调处理器
        callback_handler = StreamlitCallbackHandler(message_placeholder)

        # 初始化模型时传入回调处理器
        chat = initialize_chat_model(streaming_callback=callback_handler)
        if not chat:
            return False

        # 构建消息历史
        messages = [
            ("system", "你是一个有用的AI助手，请用友好、专业的语气回答用户问题。回答要简洁明了。")
        ]

        # 添加上下文消息（最近5轮对话）
        recent_messages = st.session_state.messages[-10:]  # 限制上下文长度
        for msg in recent_messages:
            if msg["role"] == "user":
                messages.append(("human", msg["content"]))
            else:
                messages.append(("ai", msg["content"]))

        # 添加当前用户输入
        messages.append(("human", user_input))

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages(messages)

        # 方法1: 使用 invoke 方法（推荐）
        chain = prompt | chat

        # 调用模型（流式）
        response = chain.invoke({})

        # 返回完整的响应内容
        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        message_placeholder.error(f"请求失败: {str(e)}")
        return False


# 备选方案：使用更简单的方法
def get_streaming_response_simple(user_input, message_placeholder):
    """简化版本的流式响应"""
    try:
        # 创建回调处理器
        callback_handler = StreamlitCallbackHandler(message_placeholder)

        # 初始化模型
        chat = ChatZhipuAI(
            model="glm-4",
            api_key=os.getenv('ZHIPUAI_API_KEY') or st.secrets.get("ZHIPUAI_API_KEY", ""),
            temperature=0.7,
            top_p=0.9,
            streaming=True,
            callbacks=[callback_handler],  # 在某些版本中这样使用
        )

        if not chat:
            return False

        # 构建消息列表
        message_list = []

        # 添加上下文
        recent_messages = st.session_state.messages[-8:]
        for msg in recent_messages:
            if msg["role"] == "user":
                message_list.append(("human", msg["content"]))
            else:
                message_list.append(("ai", msg["content"]))

        # 添加当前消息
        message_list.append(("human", user_input))

        # 创建提示
        prompt = ChatPromptTemplate.from_messages(
            [("system", "你是一个有用的AI助手")] + message_list
        )

        # 使用 LCEL (LangChain Expression Language)
        chain = prompt | chat

        # 调用链
        response = chain.invoke({})

        return response.content

    except Exception as e:
        message_placeholder.error(f"请求失败: {str(e)}")
        return False


# 最终解决方案：使用最兼容的方式
def get_streaming_response_final(user_input, message_placeholder):
    """最终兼容版本的流式响应"""
    try:
        api_key = os.getenv('ZHIPUAI_API_KEY') or st.secrets.get("ZHIPUAI_API_KEY", "")
        if not api_key:
            message_placeholder.error("请先设置API密钥")
            return False

        # 创建回调处理器
        callback_handler = StreamlitCallbackHandler(message_placeholder)

        # 直接初始化模型
        llm = ChatZhipuAI(
            model="glm-4",
            api_key=api_key,
            temperature=0.7,
            top_p=0.9,
            streaming=True,
            callbacks=[callback_handler],
        )

        # 构建对话历史
        conversation_history = []
        for msg in st.session_state.messages[-6:]:  # 限制历史长度
            if msg["role"] == "user":
                conversation_history.append(f"用户: {msg['content']}")
            else:
                conversation_history.append(f"助手: {msg['content']}")

        # 构建完整的提示
        if conversation_history:
            history_text = "\n".join(conversation_history)
            full_prompt = f"{history_text}\n用户: {user_input}\n助手: "
        else:
            full_prompt = f"用户: {user_input}\n助手: "

        # 直接调用模型
        response = llm.invoke(full_prompt)

        return response.content

    except Exception as e:
        message_placeholder.error(f"请求失败: {str(e)}")
        return False


# 页面标题和描述
st.title("🤖 AI聊天机器人")
st.markdown("---")
st.markdown("体验实时对话，感受更自然的交流方式！")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")

    # API密钥输入
    if not os.getenv('ZHIPUAI_API_KEY'):
        api_key_input = st.text_input(
            "智谱AI API密钥",
            type="password",
            placeholder="在此输入您的API密钥",
            help="您可以在智谱AI开放平台获取API密钥"
        )
        if api_key_input:
            os.environ['ZHIPUAI_API_KEY'] = api_key_input
            st.success("✅ API密钥已设置")

    # 响应方法选择
    st.subheader("响应模式")
    response_mode = st.radio(
        "选择响应方式",
        ["简化模式", "兼容模式"],
        index=1,
        help="如果一种模式不工作，请尝试另一种"
    )

    st.markdown("---")
    st.subheader("💡 使用提示")
    st.markdown("""
    - 💬 输入问题，体验实时回答
    - ⏳ 长回答会有明显的打字机效果
    - 🔄 支持多轮对话上下文
    - 🗑️ 可随时清空对话历史
    """)

    # 清除对话按钮
    if st.button("🗑️ 清除对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 显示聊天消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息到会话状态
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示AI回复（流式）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # 根据选择的模式调用不同的方法
        if response_mode == "简化模式":
            full_response = get_streaming_response_simple(prompt, message_placeholder)
        else:
            full_response = get_streaming_response_final(prompt, message_placeholder)

        if full_response:
            # 确保消息完全显示
            message_placeholder.markdown(full_response)
            # 添加AI回复到会话状态
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# 底部信息
st.markdown("---")
st.caption("🚀 Powered by LangChain 1.0.4 + 智谱AI GLM-4 + Streamlit")
st.caption("✨ 实时输出 | 多轮对话 | 兼容性优化")

# 空状态提示
if not st.session_state.messages:
    st.info("👆 在上方输入框开始您的对话，体验流式回答效果！")