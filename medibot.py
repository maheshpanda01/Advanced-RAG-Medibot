from dotenv import load_dotenv
import streamlit as st
import requests

load_dotenv()

st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🧠")

st.title("🧠 Medical Assistant")
st.markdown("Ask any medical question. Answers are sourced from your uploaded PDF.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Assistant:** {msg['content']}")

query = st.text_input("💬 Your question:", key="query")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    API_URL = "http://127.0.0.1:8000/ask"

    with st.spinner("🧠 Generating response..."):
        response = requests.post(API_URL, json={"query": query})

        answer = response.json()["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.markdown(f"**🤖 Assistant:** {answer}")

if st.button("🗑 Clear Chat History"):
    for key in ["messages", "query"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()




# query = st.text_input("💬 Your question:", key="query")

# if query:
#     st.session_state.messages.append({"role": "user", "content": query})

#     query_clean = query.lower().strip()
#     exact_key = f"exact:{query_clean}"

#     # -------------------------
#     # 1️⃣ EXACT CACHE CHECK
#     # -------------------------
#     cached_exact = redis_client.get(exact_key)

#     if cached_exact:
#         response_text = json.loads(cached_exact)
#         st.info("⚡ Answer from exact cache")

#     else:

#         # -------------------------
#         # 2️⃣ SEMANTIC CACHE CHECK
#         # -------------------------
#         query_embedding = embedding_model.embed_query(query)

#         cached_response = None
#         keys = redis_client.keys("semantic:*")

#         for key in keys:

#             cached_data = json.loads(redis_client.get(key))

#             cached_embedding = np.array(cached_data["embedding"]).reshape(1, -1)
#             query_embedding_np = np.array(query_embedding).reshape(1, -1)

#             similarity = cosine_similarity(query_embedding_np, cached_embedding)[0][0]

#             if similarity > 0.90:
#                 cached_response = cached_data["response"]
#                 st.info("⚡ Answer from semantic cache")
#                 break

#         if cached_response:
#             response_text = cached_response

#         else:

#             # -------------------------
#             # 3️⃣ RUN RAG PIPELINE
#             # -------------------------
#             with st.spinner("🔍 Retrieving relevant context..."):
#                 search_results = compression_retriever.invoke(query)

#             context = "\n\n".join([
#                 f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'Unknown')}"
#                 for result in search_results
#             ])

#             system_prompt = f"""
# You are a helpful AI Medical Assistant.
# Your answers must be based ONLY on the provided context from the PDF.
# always include the correct page numbers in parentheses.

# If the answer is not found in the context, reply with:
# "I could not find relevant information in the provided document."

# Context:
# {context}
# """

#             messages_for_llm = [
#                 SystemMessage(content=system_prompt),
#                 HumanMessage(content=query)
#             ]

#             with st.spinner("🧠 Generating response..."):
#                 response = llm.invoke(messages_for_llm)

#             response_text = response.content

#             # -------------------------
#             # 4️⃣ SAVE TO CACHE
#             # -------------------------
#             redis_client.set(
#                 exact_key,
#                 json.dumps(response_text)
#             )

#             redis_client.set(
#                 f"semantic:{query_clean}",
#                 json.dumps({
#                     "query": query,
#                     "embedding": query_embedding,
#                     "response": response_text
#                 })
#             )

#     # Save assistant response
#     st.session_state.messages.append({"role": "assistant", "content": response_text})

#     st.markdown(f"**🤖 Assistant:** {response_text}")

# # Clear history button
# if st.button("🗑 Clear Chat History"):
#     for key in ["messages", "query"]:
#         if key in st.session_state:
#             del st.session_state[key]
#     st.rerun()
