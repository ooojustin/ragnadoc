from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from ragnadoc.docs import DocInfo
import logging

class DocumentProcessor:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len  # type: ignore
    ):
        self.logger = logging.getLogger(__name__)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[" ", "\n", ""]
        )

    def _clean_text(self, text: str) -> str:
        cleaned = text.replace('\r\n', '\n')
        return '\n'.join(line.rstrip() for line in cleaned.split('\n')).strip()

    def process(self, doc: DocInfo) -> List[Document]:
        try:
            cleaned_content = self._clean_text(doc.content)

            # prepare base metadata
            metadata = {
                "source": doc.path,
                "repo": doc.repo_name,
                "sha": doc.sha,
                "last_modified": doc.last_modified.isoformat() if doc.last_modified else None,
                "author": doc.author,
                "document_type": "markdown" if doc.path.endswith(".md") else "mdx",
            }

            # Placeholder for empty values
            metadata = {
                key: (value if value is not None else "N/A")
                for key, value in metadata.items()
            }

            # split the document
            chunks = self.text_splitter.create_documents(
                texts=[cleaned_content],
                metadatas=[metadata]
            )

            if not chunks:
                self.logger.warning(f"No chunks created for document: {doc.path}")
                return []

            # add chunk information to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                })

            avg_size = sum(len(c.page_content) for c in chunks) / len(chunks)
            self.logger.debug(
                f"Processed {doc.path}: created {len(chunks)} chunks "
                f"(avg size: {avg_size:.0f} chars)"
            )

            return chunks

        except Exception as e:
            self.logger.error(f"Error processing document {doc.path}: {str(e)}")