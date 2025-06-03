#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import datetime
import json
import logging
import re
from collections import defaultdict

import json_repair

from api import settings
from api.db import LLMType
from rag.settings import TAG_FLD
from rag.utils import encoder, num_tokens_from_string


def chunks_format(reference):
    def get_value(d, k1, k2):
        return d.get(k1, d.get(k2))

    return [
        {
            "id": get_value(chunk, "chunk_id", "id"),
            "content": get_value(chunk, "content", "content_with_weight"),
            "document_id": get_value(chunk, "doc_id", "document_id"),
            "document_name": get_value(chunk, "docnm_kwd", "document_name"),
            "dataset_id": get_value(chunk, "kb_id", "dataset_id"),
            "image_id": get_value(chunk, "image_id", "img_id"),
            "positions": get_value(chunk, "positions", "position_int"),
            "url": chunk.get("url"),
            "similarity": chunk.get("similarity"),
            "vector_similarity": chunk.get("vector_similarity"),
            "term_similarity": chunk.get("term_similarity"),
            "doc_type": chunk.get("doc_type_kwd"),
        }
        for chunk in reference.get("chunks", [])
    ]


def llm_id2llm_type(llm_id):
    from api.db.services.llm_service import TenantLLMService

    llm_id, *_ = TenantLLMService.split_model_name_and_factory(llm_id)

    llm_factories = settings.FACTORY_LLM_INFOS
    for llm_factory in llm_factories:
        for llm in llm_factory["llm"]:
            if llm_id == llm["llm_name"]:
                return llm["model_type"].strip(",")[-1]


def message_fit_in(msg, max_length=4000):
    def count():
        nonlocal msg
        tks_cnts = []
        for m in msg:
            tks_cnts.append({"role": m["role"], "count": num_tokens_from_string(m["content"])})
        total = 0
        for m in tks_cnts:
            total += m["count"]
        return total

    c = count()
    if c < max_length:
        return c, msg

    msg_ = [m for m in msg if m["role"] == "system"]
    if len(msg) > 1:
        msg_.append(msg[-1])
    msg = msg_
    c = count()
    if c < max_length:
        return c, msg

    ll = num_tokens_from_string(msg_[0]["content"])
    ll2 = num_tokens_from_string(msg_[-1]["content"])
    if ll / (ll + ll2) > 0.8:
        m = msg_[0]["content"]
        m = encoder.decode(encoder.encode(m)[: max_length - ll2])
        msg[0]["content"] = m
        return max_length, msg

    m = msg_[-1]["content"]
    m = encoder.decode(encoder.encode(m)[: max_length - ll2])
    msg[-1]["content"] = m
    return max_length, msg


def kb_prompt(kbinfos, max_tokens):
    from api.db.services.document_service import DocumentService

    knowledges = [ck["content_with_weight"] for ck in kbinfos["chunks"]]
    used_token_count = 0
    chunks_num = 0
    for i, c in enumerate(knowledges):
        used_token_count += num_tokens_from_string(c)
        chunks_num += 1
        if max_tokens * 0.97 < used_token_count:
            knowledges = knowledges[:i]
            logging.warning(f"Not all the retrieval into prompt: {i + 1}/{len(knowledges)}")
            break

    docs = DocumentService.get_by_ids([ck["doc_id"] for ck in kbinfos["chunks"][:chunks_num]])
    docs = {d.id: d.meta_fields for d in docs}

    doc2chunks = defaultdict(lambda: {"chunks": [], "meta": []})
    for i, ck in enumerate(kbinfos["chunks"][:chunks_num]):
        cnt = f"---\nID: {i}\n" + (f"URL: {ck['url']}\n" if "url" in ck else "")
        cnt += ck["content_with_weight"]
        doc2chunks[ck["docnm_kwd"]]["chunks"].append(cnt)
        doc2chunks[ck["docnm_kwd"]]["meta"] = docs.get(ck["doc_id"], {})

    knowledges = []
    for nm, cks_meta in doc2chunks.items():
        txt = f"\nDocument: {nm} \n"
        for k, v in cks_meta["meta"].items():
            txt += f"{k}: {v}\n"
        txt += "Relevant fragments as following:\n"
        for i, chunk in enumerate(cks_meta["chunks"], 1):
            txt += f"{chunk}\n"
        knowledges.append(txt)
    return knowledges


def citation_prompt():
    print("USE PROMPT", flush=True)
    return """

# Citation requirements:

- Use a uniform citation format of like [ID:i] [ID:j], where "i" and "j" are the document ID enclosed in square brackets. Separate multiple IDs with spaces (e.g., [ID:0] [ID:1]).
- Citation markers must be placed at the end of a sentence, separated by a space from the final punctuation (e.g., period, question mark). A maximum of 4 citations are allowed per sentence.
- DO NOT insert CITATION in the answer if the content is not from retrieved chunks.
- DO NOT use standalone Document IDs (e.g., '#ID#').
- Citations ALWAYS in the "[ID:i]" format.
- STRICTLY prohibit the use of strikethrough symbols (e.g., ~~) or any other non-standard formatting syntax.
- Any failure to adhere to the above rules, including but not limited to incorrect formatting, use of prohibited styles, or unsupported citations, will be considered an error, and no citation will be added for that sentence.

--- Example START ---
<SYSTEM>: Here is the knowledge base:

Document: Elon Musk Breaks Silence on Crypto, Warns Against Dogecoin ...
URL: https://blockworks.co/news/elon-musk-crypto-dogecoin
ID: 0
The Tesla co-founder advised against going all-in on dogecoin, but Elon Musk said it’s still his favorite crypto...

Document: Elon Musk's Dogecoin tweet sparks social media frenzy
ID: 1
Musk said he is 'willing to serve' D.O.G.E. – shorthand for Dogecoin.

Document: Causal effect of Elon Musk tweets on Dogecoin price
ID: 2
If you think of Dogecoin — the cryptocurrency based on a meme — you can’t help but also think of Elon Musk...

Document: Elon Musk's Tweet Ignites Dogecoin's Future In Public Services
ID: 3
The market is heating up after Elon Musk's announcement about Dogecoin. Is this a new era for crypto?...

      The above is the knowledge base.

<USER>: What's the Elon's view on dogecoin?

<ASSISTANT>: Musk has consistently expressed his fondness for Dogecoin, often citing its humor and the inclusion of dogs in its branding. He has referred to it as his favorite cryptocurrency [ID:0] [ID:1].
Recently, Musk has hinted at potential future roles for Dogecoin. His tweets have sparked speculation about Dogecoin's potential integration into public services [ID:3].
Overall, while Musk enjoys Dogecoin and often promotes it, he also warns against over-investing in it, reflecting both his personal amusement and caution regarding its speculative nature.

--- Example END ---

"""

'''
--- disable origin code for enhance keyword extraction --- Mage 6/3/2025 

def keyword_extraction(chat_mdl, content, topn=3):
    prompt = f"""
Role: You're a text analyzer.
Task: extract the most important keywords/phrases of a given piece of text content.
Requirements:
  - Summarize the text content, and give top {topn} important keywords/phrases.
  - The keywords MUST be in language of the given piece of text content.
  - The keywords are delimited by ENGLISH COMMA.
  - Keywords ONLY in output.

### Text Content
{content}

"""
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd
'''
# new function for extract PIMS data

def keyword_extraction(chat_mdl, content, topn=3):
    """
    Enhanced keyword extraction for technical documents.
    Maintains compatibility with original function signature.
    Automatically extracts PIMS numbers, project phases, and project names.
    """
    
    # 預先提取結構化資訊
    structured_info = _extract_structured_info(content)
    
    # 構建增強的提示詞
    prompt = f"""
Role: You're a text analyzer specializing in technical and engineering documents.
Task: extract the most important keywords/phrases of a given piece of text content.
Requirements:
  - Summarize the text content, and give top {topn} important keywords/phrases.
  - PRIORITY: Always include PIMS numbers (PIMS-XXXXXX), project phases (DVT1/DVT2/MP/EVT/PVT), and project names if found.
  - Focus on technical components, issues, and engineering terms.
  - The keywords MUST be in language of the given piece of text content.
  - The keywords are delimited by ENGLISH COMMA.
  - Keywords ONLY in output.

### Text Content
{content}
"""

    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    
    if kwd.find("**ERROR**") >= 0:
        # 回退到結構化提取
        fallback_keywords = []
        if structured_info["pims_no"]:
            fallback_keywords.extend(structured_info["pims_no"])
        if structured_info["project_phase"]:
            fallback_keywords.extend(structured_info["project_phase"])
        if structured_info["project_name"]:
            fallback_keywords.extend(structured_info["project_name"])
        return ", ".join(fallback_keywords) if fallback_keywords else ""
    
    # 後處理：確保結構化資訊被包含
    llm_keywords = [k.strip() for k in kwd.split(",") if k.strip()]
    final_keywords = []
    
    # 優先添加結構化資訊
    if structured_info["pims_no"]:
        final_keywords.extend(structured_info["pims_no"])
    if structured_info["project_phase"]:
        final_keywords.extend(structured_info["project_phase"])
    if structured_info["project_name"]:
        final_keywords.extend(structured_info["project_name"])
    
    # 添加 LLM 提取的關鍵詞，避免重複
    for keyword in llm_keywords:
        if keyword.lower() not in [k.lower() for k in final_keywords]:
            final_keywords.append(keyword)
    
    # 如果沒有結構化資訊，直接返回 LLM 結果
    if not any([structured_info["pims_no"], structured_info["project_phase"], structured_info["project_name"]]):
        return kwd
    
    # 返回合併後的關鍵詞
    max_keywords = topn + len(structured_info["pims_no"]) + len(structured_info["project_phase"]) + len(structured_info["project_name"])
    return ", ".join(final_keywords[:max_keywords])


def _extract_structured_info(content):
    """
    Private helper function to extract structured information using regex patterns.
    Prefix with underscore to indicate internal use.
    """
    structured_info = {
        "pims_no": [],
        "project_phase": [],
        "project_name": []
    }
    
    # 1. 提取 PIMS 編號
    pims_patterns = [
        r'PIMS[-\s]?(\d{6})',  # PIMS-315392, PIMS 315392, PIMS315392
        r'PIMS[-\s]?(\d{5})',  # PIMS-23095, PIMS 23095
    ]
    
    for pattern in pims_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            pims_full = f"PIMS-{match}"
            if pims_full not in structured_info["pims_no"]:
                structured_info["pims_no"].append(pims_full)
    
    # 2. 提取專案階段
    phase_patterns = [
        r'\b(DVT\d+(?:\.\d+)?)\b',  # DVT1, DVT1.0, DVT2, etc.
        r'\b(EVT\d*)\b',            # EVT, EVT1, EVT2
        r'\b(PVT\d*)\b',            # PVT, PVT1, PVT2
        r'\b(MP\d*)\b',             # MP, MP1, MP2
        r'\b(proto\d*)\b',          # proto, proto1
        r'\b(pilot\d*)\b',          # pilot, pilot1
    ]
    
    for pattern in phase_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if match.upper() not in [p.upper() for p in structured_info["project_phase"]]:
                structured_info["project_phase"].append(match.upper())
    
    # 3. 提取專案名稱
    project_name_patterns = [
        r'\b([A-Z][a-z]+\d+)\b',     # Sanctuary18, KDF60
        r'\b([A-Z]{3,}[\d]*)\b',     # KDK, KDF
        r'\b([A-Z][a-z]{2,})\b',     # Sanctuary
    ]
    
    # 從標題或特定上下文中提取
    lines = content.split('\n')
    for line in lines[:10]:  # 檢查前10行
        line = line.strip()
        if 'PIMS' in line or any(phase in line.upper() for phase in ['DVT', 'EVT', 'PVT', 'MP']):
            for pattern in project_name_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    exclude_words = ['Dell', 'Customer', 'Communication', 'Confidential', 
                                   'Global', 'Marketing', 'DVT', 'EVT', 'PVT', 'PIMS']
                    if (match not in exclude_words and 
                        match.lower() not in [p.lower() for p in structured_info["project_name"]] and
                        len(match) >= 3):
                        structured_info["project_name"].append(match)
    
    return structured_info


def question_proposal(chat_mdl, content, topn=3):
    prompt = f"""
Role: You're a text analyzer.
Task:  propose {topn} questions about a given piece of text content.
Requirements:
  - Understand and summarize the text content, and propose top {topn} important questions.
  - The questions SHOULD NOT have overlapping meanings.
  - The questions SHOULD cover the main content of the text as much as possible.
  - The questions MUST be in language of the given piece of text content.
  - One question per line.
  - Question ONLY in output.

### Text Content
{content}

"""
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


def full_question(tenant_id, llm_id, messages, language=None):
    from api.db.services.llm_service import LLMBundle

    if llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    conv = []
    for m in messages:
        if m["role"] not in ["user", "assistant"]:
            continue
        conv.append("{}: {}".format(m["role"].upper(), m["content"]))
    conv = "\n".join(conv)
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    prompt = f"""
Role: A helpful assistant

Task and steps:
    1. Generate a full user question that would follow the conversation.
    2. If the user's question involves relative date, you need to convert it into absolute date based on the current date, which is {today}. For example: 'yesterday' would be converted to {yesterday}.

Requirements & Restrictions:
  - If the user's latest question is completely, don't do anything, just return the original question.
  - DON'T generate anything except a refined question."""
    if language:
        prompt += f"""
  - Text generated MUST be in {language}."""
    else:
        prompt += """
  - Text generated MUST be in the same language of the original user's question.
"""
    prompt += f"""

######################
-Examples-
######################

# Example 1
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
###############
Output: What's the name of Donald Trump's mother?

------------
# Example 2
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
ASSISTANT:  Mary Trump.
User: What's her full name?
###############
Output: What's the full name of Donald Trump's mother Mary Trump?

------------
# Example 3
## Conversation
USER: What's the weather today in London?
ASSISTANT:  Cloudy.
USER: What's about tomorrow in Rochester?
###############
Output: What's the weather in Rochester on {tomorrow}?

######################
# Real Data
## Conversation
{conv}
###############
    """
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": "Output: "}], {"temperature": 0.2})
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
    return ans if ans.find("**ERROR**") < 0 else messages[-1]["content"]


def cross_languages(tenant_id, llm_id, query, languages=[]):
    from api.db.services.llm_service import LLMBundle

    if llm_id and llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)

    sys_prompt = """
Act as a streamlined multilingual translator. Strictly output translations separated by ### without any explanations or formatting. Follow these rules:

1. Accept batch translation requests in format:
[source text]
=== 
[target languages separated by commas]

2. Always maintain:
- Original formatting (tables/lists/spacing)
- Technical terminology accuracy
- Cultural context appropriateness

3. Output format:
[language1 translation] 
### 
[language1 translation]

**Examples:**
Input:
Hello World! Let's discuss AI safety.
===
Chinese, French, Jappanese

Output:
你好世界！让我们讨论人工智能安全问题。
###
Bonjour le monde ! Parlons de la sécurité de l'IA.
###
こんにちは世界！AIの安全性について話し合いましょう。
"""
    user_prompt = f"""
Input:
{query}
===
{", ".join(languages)}

Output:
"""

    ans = chat_mdl.chat(sys_prompt, [{"role": "user", "content": user_prompt}], {"temperature": 0.2})
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
    if ans.find("**ERROR**") >= 0:
        return query
    return "\n".join([a for a in re.sub(r"(^Output:|\n+)", "", ans, flags=re.DOTALL).split("===") if a.strip()])


''' 
--- disable for enhance content tagging ---

def content_tagging(chat_mdl, content, all_tags, examples, topn=3):
    prompt = f"""
Role: You're a text analyzer.

Task: Tag (put on some labels) to a given piece of text content based on the examples and the entire tag set.

Steps::
  - Comprehend the tag/label set.
  - Comprehend examples which all consist of both text content and assigned tags with relevance score in format of JSON.
  - Summarize the text content, and tag it with top {topn} most relevant tags from the set of tag/label and the corresponding relevance score.

Requirements
  - The tags MUST be from the tag set.
  - The output MUST be in JSON format only, the key is tag and the value is its relevance score.
  - The relevance score must be range from 1 to 10.
  - Keywords ONLY in output.

# TAG SET
{", ".join(all_tags)}

"""
    for i, ex in enumerate(examples):
        prompt += """
# Examples {}
### Text Content
{}

Output:
{}

        """.format(i, ex["content"], json.dumps(ex[TAG_FLD], indent=2, ensure_ascii=False))

    prompt += f"""
# Real Data
### Text Content
{content}

"""
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.5})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        raise Exception(kwd)

    try:
        obj = json_repair.loads(kwd)
    except json_repair.JSONDecodeError:
        try:
            result = kwd.replace(prompt[:-1], "").replace("user", "").replace("model", "").strip()
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            obj = json_repair.loads(result)
        except Exception as e:
            logging.exception(f"JSON parsing error: {result} -> {e}")
            raise e
    res = {}
    for k, v in obj.items():
        try:
            res[str(k)] = int(v)
        except Exception:
            pass
    return res


'''

# new function - content tagging with content and file name - by Mage - 6/3/2025

def content_tagging(chat_mdl, content, all_tags, examples, topn=3):
    """
    為文本內容進行標籤分配，自動整合檔名信息進行分析
    
    Args:
        chat_mdl: 聊天模型實例
        content: 要打標籤的文本內容
        all_tags: 所有可用的標籤集合
        examples: 包含範例文本和對應標籤的示例數據
        topn: 要返回的最相關標籤數量，預設為3個
    
    Returns:
        dict: 標籤和相關性分數的字典
    """
    
    # 嘗試從 content 或上下文中提取檔名信息
    filename_info = _extract_filename_from_context(content)
    
    # 準備分析的文本
    analysis_text = content
    has_filename = False
    
    # 如果找到檔名信息，將其整合到分析中
    if filename_info:
        has_filename = True
        analysis_text = f"Document filename: {filename_info}\n\nDocument content:\n{content}"
    
    prompt = f"""
Role: You're a text analyzer.

Task: Tag (put on some labels) to a given piece of text content based on the examples and the entire tag set.

Steps::
  - Comprehend the tag/label set.
  - Comprehend examples which all consist of both text content and assigned tags with relevance score in format of JSON.
  - Summarize the text content, and tag it with top {topn} most relevant tags from the set of tag/label and the corresponding relevance score.

Requirements
  - The tags MUST be from the tag set.
  - The output MUST be in JSON format only, the key is tag and the value is its relevance score.
  - The relevance score must be range from 1 to 10.
  - Keywords ONLY in output."""
    
    # 如果有檔名信息，在要求中加入說明
    if has_filename:
        prompt += """
  - Consider both the document filename and content when assigning tags.
  - The filename may provide additional context about the document's topic or category."""

    prompt += f"""

# TAG SET
{", ".join(all_tags)}

"""
    for i, ex in enumerate(examples):
        prompt += """
# Examples {}
### Text Content
{}

Output:
{}

        """.format(i, ex["content"], json.dumps(ex[TAG_FLD], indent=2, ensure_ascii=False))

    prompt += f"""
# Real Data
### Text Content
{analysis_text}

"""
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.5})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        raise Exception(kwd)

    try:
        obj = json_repair.loads(kwd)
    except json_repair.JSONDecodeError:
        try:
            result = kwd.replace(prompt[:-1], "").replace("user", "").replace("model", "").strip()
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            obj = json_repair.loads(result)
        except Exception as e:
            logging.exception(f"JSON parsing error: {result} -> {e}")
            raise e
    res = {}
    for k, v in obj.items():
        try:
            res[str(k)] = int(v)
        except Exception:
            pass
    return res


def _extract_filename_from_context(content):
    """
    從內容或可能的上下文中提取檔名信息
    這個函數會嘗試多種方式來獲取檔名，確保不影響原有功能
    
    Args:
        content: 文本內容
        
    Returns:
        str: 清理後的檔名信息，如果沒有找到則返回None
    """
    import re
    import os
    import inspect
    
    # 方法1: 檢查是否在內容中已經包含檔名信息
    filename_patterns = [
        r'filename[:\s]+([^\n\r]+)',
        r'file[:\s]+([^\n\r]+\.[a-zA-Z]{2,4})',
        r'document[:\s]+([^\n\r]+\.[a-zA-Z]{2,4})',
    ]
    
    for pattern in filename_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return _clean_filename(match.group(1).strip())
    
    # 方法2: 嘗試從調用堆疊中獲取文檔相關信息
    try:
        frame = inspect.currentframe()
        while frame:
            frame = frame.f_back
            if not frame:
                break
                
            # 檢查局部變量中是否有文檔相關信息
            local_vars = frame.f_locals
            
            # 常見的變檔名變量名
            filename_vars = ['filename', 'file_name', 'doc_name', 'document_name', 
                           'name', 'fname', 'file', 'document']
            
            for var_name in filename_vars:
                if var_name in local_vars:
                    value = local_vars[var_name]
                    if isinstance(value, str) and value.strip():
                        # 檢查是否看起來像檔名
                        if '.' in value or len(value) > 3:
                            return _clean_filename(value)
            
            # 檢查是否有文檔對象
            doc_vars = ['doc', 'd', 'document', 'bx', 'chunk']
            for var_name in doc_vars:
                if var_name in local_vars:
                    doc_obj = local_vars[var_name]
                    if hasattr(doc_obj, 'get'):
                        # 嘗試從字典或對象中獲取檔名
                        possible_names = ['name', 'filename', 'file_name', 'docnm_kwd', 'title']
                        for name_key in possible_names:
                            if isinstance(doc_obj, dict) and name_key in doc_obj:
                                filename = doc_obj[name_key]
                                if isinstance(filename, str) and filename.strip():
                                    return _clean_filename(filename)
                    elif hasattr(doc_obj, 'name'):
                        if isinstance(doc_obj.name, str) and doc_obj.name.strip():
                            return _clean_filename(doc_obj.name)
                            
    except Exception:
        # 如果在檢查過程中出現任何錯誤，忽略並繼續
        pass
    
    return None


def _clean_filename(filename):
    """
    清理檔名，提取有用的信息
    
    Args:
        filename: 原始檔名
        
    Returns:
        str: 清理後的檔名信息
    """
    import os
    import re
    
    if not filename or not isinstance(filename, str):
        return None
        
    filename = filename.strip()
    if not filename:
        return None
    
    # 去除路徑和副檔名
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # 如果檔名太短或看起來不像有意義的名稱，返回None
    if len(base_filename) < 2:
        return None
        
    # 清理檔名：替換常見分隔符為空格
    clean_filename = base_filename.replace('_', ' ').replace('-', ' ').replace('.', ' ')
    
    # 移除多餘的空格
    clean_filename = re.sub(r'\s+', ' ', clean_filename).strip()
    
    # 如果清理後的檔名太短，返回None
    if len(clean_filename) < 3:
        return None
    
    return clean_filename


def _enhance_examples_with_filename(examples):
    """
    為範例數據添加檔名信息（如果可用）
    這個函數會嘗試從範例中提取檔名信息並格式化
    
    Args:
        examples: 原始範例數據
        
    Returns:
        list: 可能包含檔名信息的範例數據
    """
    enhanced_examples = []
    
    for ex in examples:
        enhanced_ex = ex.copy()
        
        # 檢查範例中是否有檔名相關信息
        filename_keys = ['filename', 'file_name', 'doc_name', 'document_name', 'name']
        filename_found = None
        
        for key in filename_keys:
            if key in ex and isinstance(ex[key], str) and ex[key].strip():
                filename_found = _clean_filename(ex[key])
                break
        
        # 如果找到檔名，將其加入到內容中
        if filename_found:
            enhanced_ex["content"] = f"Document filename: {filename_found}\n\nDocument content:\n{ex['content']}"
        
        enhanced_examples.append(enhanced_ex)
    
    return enhanced_examples

def vision_llm_describe_prompt(page=None) -> str:
    prompt_en = """
INSTRUCTION:
Transcribe the content from the provided PDF page image into clean Markdown format.
- Only output the content transcribed from the image.
- Do NOT output this instruction or any other explanation.
- If the content is missing or you do not understand the input, return an empty string.

RULES:
1. Do NOT generate examples, demonstrations, or templates.
2. Do NOT output any extra text such as 'Example', 'Example Output', or similar.
3. Do NOT generate any tables, headings, or content that is not explicitly present in the image.
4. Transcribe content word-for-word. Do NOT modify, translate, or omit any content.
5. Do NOT explain Markdown or mention that you are using Markdown.
6. Do NOT wrap the output in ```markdown or ``` blocks.
7. Only apply Markdown structure to headings, paragraphs, lists, and tables, strictly based on the layout of the image. Do NOT create tables unless an actual table exists in the image.
8. Preserve the original language, information, and order exactly as shown in the image.
"""

    if page is not None:
        prompt_en += f"\nAt the end of the transcription, add the page divider: `--- Page {page} ---`."

    prompt_en += """
FAILURE HANDLING:
- If you do not detect valid content in the image, return an empty string.
"""
    return prompt_en


def vision_llm_figure_describe_prompt() -> str:
    prompt = """
You are an expert visual data analyst. Analyze the image and provide a comprehensive description of its content. Focus on identifying the type of visual data representation (e.g., bar chart, pie chart, line graph, table, flowchart), its structure, and any text captions or labels included in the image.

Tasks:
1. Describe the overall structure of the visual representation. Specify if it is a chart, graph, table, or diagram.
2. Identify and extract any axes, legends, titles, or labels present in the image. Provide the exact text where available.
3. Extract the data points from the visual elements (e.g., bar heights, line graph coordinates, pie chart segments, table rows and columns).
4. Analyze and explain any trends, comparisons, or patterns shown in the data.
5. Capture any annotations, captions, or footnotes, and explain their relevance to the image.
6. Only include details that are explicitly present in the image. If an element (e.g., axis, legend, or caption) does not exist or is not visible, do not mention it.

Output format (include only sections relevant to the image content):
- Visual Type: [Type]
- Title: [Title text, if available]
- Axes / Legends / Labels: [Details, if available]
- Data Points: [Extracted data]
- Trends / Insights: [Analysis and interpretation]
- Captions / Annotations: [Text and relevance, if available]

Ensure high accuracy, clarity, and completeness in your analysis, and includes only the information present in the image. Avoid unnecessary statements about missing elements.
"""
    return prompt
