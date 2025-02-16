import argparse
import logging
import os
import time
from threading import Timer

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

# Parse arguments before importing assistant modules.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default="configs/demo_hospital.yaml",
    help="Path to tenant config file",
)
args, _ = parser.parse_known_args()
os.environ["TENANT_CONFIG_PATH"] = args.config

from groqflow.config import UPLOADED_DOCUMENTS_PATH, VECTOR_DB_PATH
from groqflow.knowledge_base import KnowledgeBase

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FolderMonitor(FileSystemEventHandler):
    def __init__(self, folder_path: str, knowledge_base: KnowledgeBase) -> None:
        self.folder_path = folder_path
        self.knowledge_base = knowledge_base
        self._timers: dict[str, Timer] = {}
        self.debounce_seconds = 2.0

    def _process_file(self, file_path: str) -> None:
        logging.info(f"Processing file: {file_path}")
        try:
            self.knowledge_base.update(file_path)
        except Exception as exc:
            logging.error(f"Error updating knowledge base with {file_path}: {exc}")

    def _debounce(self, file_path: str) -> None:
        if file_path in self._timers:
            self._timers[file_path].cancel()
        timer = Timer(self.debounce_seconds, self._process_file, args=[file_path])
        self._timers[file_path] = timer
        timer.start()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._debounce(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._debounce(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            logging.info(f"File moved from {event.src_path} to {event.dest_path}")
            self.on_deleted(event)
            self._debounce(event.dest_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        file_path = event.src_path
        if file_path in self._timers:
            self._timers[file_path].cancel()
            del self._timers[file_path]

        logging.info(f"File deleted: {file_path}")
        try:
            self.knowledge_base.delete(file_path)
        except Exception as exc:
            logging.error(f"Error deleting {file_path} from knowledge base: {exc}")


def main() -> None:
    os.makedirs(UPLOADED_DOCUMENTS_PATH, exist_ok=True)

    knowledge_base = KnowledgeBase(
        folder_path=UPLOADED_DOCUMENTS_PATH,
        vector_db_path=VECTOR_DB_PATH,
    )

    event_handler = FolderMonitor(UPLOADED_DOCUMENTS_PATH, knowledge_base)
    observer = Observer()
    observer.schedule(event_handler, UPLOADED_DOCUMENTS_PATH, recursive=True)
    observer.start()
    logging.info(f"Monitoring {UPLOADED_DOCUMENTS_PATH} for document changes...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    finally:
        observer.join()


if __name__ == "__main__":
    main()
