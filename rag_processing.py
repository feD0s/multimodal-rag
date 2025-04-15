import os
import warnings
import io
import re
import base64
import uuid
import logging
import time

# Configure module-level logger
logger = logging.getLogger(__name__)

# Langchain and related imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage

# Other necessary libraries
from unstructured.partition.pdf import partition_pdf
from PIL import Image

# Custom exceptions for more granular error handling
class RAGProcessingError(Exception):
    """Base exception for RAG processing errors."""
    pass

class TextProcessingError(RAGProcessingError):
    """Exception for text processing failures."""
    pass

class ImageProcessingError(RAGProcessingError):
    """Exception for image processing failures."""
    pass

class ModelInferenceError(RAGProcessingError):
    """Exception for model inference failures."""
    pass

class VectorStoreError(RAGProcessingError):
    """Exception for vector store operations failures."""
    pass

# --- Text and Table Summarization ---
def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    Функция для создания суммаризации текста и таблиц с использованием модели GPT.

    Аргументы:
    texts: Список строк (тексты), которые нужно суммировать.
    tables: Список строк (таблицы), которые нужно суммировать.
    summarize_texts: Булев флаг, указывающий, нужно ли суммировать текстовые элементы.

    Возвращает:
    Два списка: text_summaries (суммаризации текстов) и table_summaries (суммаризации таблиц).
    """

    # Шаблон для запроса к модели. Задача ассистента - создать оптимизированное описание для поиска.
    prompt_text = """ Create summarization for the {element} """

    # Создаем шаблон запроса на основе строки с шаблоном
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Создаем модель для генерации суммаризаций. Устанавливаем температуру 0 для детерминированных ответов.
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # Определяем цепочку обработки запросов: сначала шаблон запроса, затем модель, затем парсер выходных данных
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []  # Список для хранения суммаризаций текстов
    table_summaries = []  # Список для хранения суммаризаций таблиц

    # Если есть текстовые элементы и требуется их суммирование
    if texts and summarize_texts:
        try:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
        except Exception as e:
            logger.error(f"Error summarizing texts: {e}")
            raise TextProcessingError(f"Error summarizing texts: {e}")
    elif texts:
        text_summaries = texts

    # Если есть таблицы, выполняем их суммирование
    if tables:
        try:
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
        except Exception as e:
            logger.error(f"Error summarizing tables: {e}")
            raise TextProcessingError(f"Error summarizing tables: {e}")

    return text_summaries, table_summaries


# --- PDF Processing ---
def extract_pdf_elements(path, fname, image_output_path):
    """
    Функция для извлечения различных элементов из PDF-файла.

    Аргументы:
    path: Строка, содержащая путь к директории PDF-файла.
    fname: Строка, содержащая имя PDF-файла.
    image_output_path: Строка, путь для сохранения извлеченных изображений.

    Возвращает:
    Список объектов типа `unstructured.documents.elements`.
    """
    try:
        return partition_pdf(
            filename=os.path.join(path, fname),
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="basic",
            max_characters=1500,
            new_after_n_chars=1200,
            combine_text_under_n_chars=900,
            image_output_dir_path=image_output_path,
        )
    except Exception as e:
        logger.error(f"Error partitioning PDF {fname}: {e}")
        raise RAGProcessingError(f"Error partitioning PDF {fname}: {e}")


def categorize_elements(raw_pdf_elements):
    """
    Функция для категоризации извлеченных элементов из PDF-файла.

    Аргументы:
    raw_pdf_elements: Список объектов типа `unstructured.documents.elements`.

    Возвращает:
    Два списка: texts (текстовые элементы) и tables (таблицы).
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables


# --- Image Processing ---
def encode_image(image_path):
    """
    Функция для кодирования изображения в формат base64.

    Аргументы:
    image_path: Строка, путь к изображению, которое нужно закодировать.

    Возвращает:
    Закодированное в формате base64 изображение в виде строки или None при ошибке.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        logger.warning(f"Image file not found at {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        raise ImageProcessingError(f"Error encoding image {image_path}: {e}")


def image_summarize(img_base64, prompt):
    """
    Функция для получения суммаризации изображения с использованием GPT модели.

    Аргументы:
    img_base64: Строка, изображение закодированное в формате base64.
    prompt: Строка, запрос для модели GPT.

    Возвращает:
    Суммаризация изображения или None при ошибке.
    """
    try:
        chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=2000)
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ]
        )
        return msg.content
    except Exception as e:
        logger.error(f"Error during image summarization: {e}")
        raise ModelInferenceError(f"Error during image summarization: {e}")


def generate_img_summaries(image_dir_path):
    """
    Функция для генерации суммаризаций изображений из указанной директории.

    Аргументы:
    image_dir_path: Строка, путь к директории с изображениями формата .jpg.

    Возвращает:
    Два списка: img_base64_list и image_summaries.
    """
    img_base64_list = []
    image_summaries = []
    prompt = """Create summarization for the image provided."""

    if not os.path.isdir(image_dir_path):
        logger.warning(f"Image directory not found: {image_dir_path}")
        return img_base64_list, image_summaries

    for img_file in sorted(os.listdir(image_dir_path)):
        if img_file.lower().endswith(".jpg"):
            img_path = os.path.join(image_dir_path, img_file)
            base64_image = encode_image(img_path)
            if base64_image:
                img_base64_list.append(base64_image)
                try:
                    summary = image_summarize(base64_image, prompt)
                    image_summaries.append(summary)
                except ModelInferenceError:
                    logger.warning(f"Failed to summarize image {img_file}")
            else:
                logger.warning(f"Failed to encode image {img_file}")

    return img_base64_list, image_summaries


# --- Retriever Creation ---
def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    Функция для создания ретривера, который может извлекать данные из разных источников.

    Аргументы:
    vectorstore: Векторное хранилище.
    text_summaries: Список суммаризаций текстов.
    texts: Список исходных текстов.
    table_summaries: Список суммаризаций таблиц.
    tables: Список исходных таблиц.
    image_summaries: Список суммаризаций изображений.
    images: Список изображений в формате base64.

    Возвращает:
    Созданный ретривер.
    """

    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        try:
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
        except Exception as e:
            logger.error(f"Error adding documents to retriever: {e}")
            raise VectorStoreError(f"Error adding documents to retriever: {e}")

    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        if images and len(image_summaries) == len(images):
            add_documents(retriever, image_summaries, images)
        elif images:
            logger.warning(f"Mismatch between image summaries ({len(image_summaries)}) and images ({len(images)}). Skipping image addition.")

    return retriever


# --- Image/Text Utilities ---
def looks_like_base64(sb):
    """
    Проверяет, выглядит ли строка как base64.
    """
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Проверяет, является ли base64 данные изображением.
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data[:12])
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Изменяет размер изображения, закодированного в формате base64.
    """
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        resized_img = img.resize(size, Image.LANCZOS)
        buffered = io.BytesIO()
        img_format = img.format if img.format else "JPEG"
        resized_img.save(buffered, format=img_format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        raise ImageProcessingError(f"Error resizing image: {e}")


def split_image_text_types(docs):
    """
    Разделяет документы на изображения и текстовые данные.
    """
    b64_images = []
    texts = []
    for doc in docs:
        page_content = doc.page_content if isinstance(doc, Document) else str(doc)

        if looks_like_base64(page_content) and is_image_data(page_content):
            try:
                resized_image = resize_base64_image(page_content, size=(1300, 600))
                b64_images.append(resized_image)
            except ImageProcessingError:
                logger.warning("Skipping image due to resize error.")
        else:
            texts.append(page_content)
    return {"images": b64_images, "texts": texts}


# --- Prompt Formatting ---
def img_prompt_func(data_dict):
    """
    Формирует запрос к модели с учетом изображений и текста.
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    text_message = {
        "type": "text",
        "text": (
            f"Вопрос пользователя: {data_dict['question']}\n\n"
            "Текст и / или таблицы:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


# --- RAG Chain Definition ---
def multi_modal_rag_chain(retriever):
    """
    Создает RAG цепочку для работы с мультимодальными запросами.
    """
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=8192)

    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain
