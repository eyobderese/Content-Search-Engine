import unittest
from Search import PDFProcessor


class TestPDFProcessor(unittest.TestCase):
    def test_process_text(self):
        processor = PDFProcessor()
        docs = ["This is a sample document", "Another document with text"]
        titles = ["Sample Title", "Another Title"]

        processor.process_text1(docs, titles)

        expected_docs = ["this is a sample document",
                         "another document with text"]
        expected_titles = ["sample title", "another title"]

        self.assertEqual(processor.processed_docs, expected_docs)
        self.assertEqual(processor.titles, expected_titles)


if __name__ == '__main__':
    unittest.main()
