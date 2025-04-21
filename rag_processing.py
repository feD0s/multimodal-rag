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
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
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
    Function for creating text and table summaries using the GPT model.

    Arguments:
    texts: List of strings (texts) to summarize.
    tables: List of strings (tables) to summarize.
    summarize_texts: Boolean flag indicating whether to summarize text elements.

    Returns:
    Two lists: text_summaries (text summaries) and table_summaries (table summaries).
    """

    # Template for the model prompt. Assistant's task is to create an optimized description for search.
    prompt_text = """ Create summarization for the {element} """

    # Create a prompt template based on the template string
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Create a model for generating summaries. Set temperature to 0 for deterministic responses.
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # Define the request processing chain: first prompt template, then model, then output parser
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []  # List for storing text summaries
    table_summaries = []  # List for storing table summaries    # If there are text elements and summarization is required
    if texts and summarize_texts:
        try:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
        except Exception as e:
            logger.error(f"Error summarizing texts: {e}")
            raise TextProcessingError(f"Error summarizing texts: {e}")
    elif texts:
        text_summaries = texts

    # If there are tables, perform their summarization
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
    Function for extracting various elements from a PDF file.

    Arguments:
    path: String containing the path to the PDF file directory.
    fname: String containing the name of the PDF file.
    image_output_path: String, path for saving extracted images.

    Returns:
    List of objects of type `unstructured.documents.elements`.
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
    Function for categorizing extracted elements from a PDF file.

    Arguments:
    raw_pdf_elements: List of objects of type `unstructured.documents.elements`.

    Returns:
    Two lists: texts (text elements) and tables (tables).
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
    Function for encoding an image to base64 format.

    Arguments:
    image_path: String, path to the image to encode.

    Returns:
    Base64 encoded image as a string or None on error.
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
    Function for getting an image summary using the GPT model.

    Arguments:
    img_base64: String, image encoded in base64 format.
    prompt: String, query for the GPT model.

    Returns:
    Image summary or None on error.
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
    Function for generating summaries of images from a specified directory.

    Arguments:
    image_dir_path: String, path to the directory with image files.

    Returns:
    Two lists: img_base64_list and image_summaries.
    """
    img_base64_list = []
    image_summaries = []
    prompt = """Create summarization for the image provided."""
    supported_formats = (".jpg", ".jpeg", ".png") # Add more formats if needed

    if not os.path.isdir(image_dir_path):
        logger.warning(f"Image directory not found: {image_dir_path}")
        return img_base64_list, image_summaries

    for img_file in sorted(os.listdir(image_dir_path)):
        # Check for multiple supported image formats
        if img_file.lower().endswith(supported_formats):
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
    Function for creating a retriever that can extract data from different sources.

    Arguments:
    vectorstore: Vector storage.
    text_summaries: List of text summaries.
    texts: List of original texts.
    table_summaries: List of table summaries.
    tables: List of original tables.
    image_summaries: List of image summaries.
    images: List of images in base64 format.

    Returns:
    The created retriever.
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
    Checks if a string looks like base64.
    """
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Checks if base64 data is an image.
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
    Resizes an image encoded in base64 format.
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
    Separates documents into images and text data.
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
    Forms a prompt for the model taking into account images and text.
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
            f"User question: {data_dict['question']}\n\n"
            "Text and/or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


# --- RAG Chain Definition ---
def multi_modal_rag_chain(retriever):
    """
    Creates a RAG chain for working with multimodal queries.
    Returns a dictionary containing the text answer and relevant images.
    """
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=8192)

    # Chain to retrieve context and split into text/images
    retrieve_and_split = retriever | RunnableLambda(split_image_text_types)

    # Chain to generate the final answer using the model
    generate_answer = (
        RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    # Parallel chain to run retrieval/splitting and answer generation
    # Passes the retrieved images through alongside the generated answer
    chain = RunnableParallel(
        {
            "context": retrieve_and_split,
            "question": RunnablePassthrough()
        }
    ) | RunnableParallel(
        {
            "answer": generate_answer,
            "images": lambda x: x["context"]["images"] # Pass images from the context
        }
    )

    return chain
