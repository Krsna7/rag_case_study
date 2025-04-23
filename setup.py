from setuptools import setup, find_packages

setup(
    name="rag-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "openai==0.28.0",
        "langchain==0.0.267",
        "PyPDF2>=3.0.1",
        "nltk>=3.8.1",
        "transformers>=4.30.2",
        "faiss-cpu>=1.7.4",
        "whoosh>=2.7.4",
        "scikit-learn>=1.2.2",
        "tqdm>=4.65.0",
        "numpy>=1.24.3",
        "requests>=2.31.0",
        "torch>=2.0.1",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Retrieval-Augmented Generation Pipeline for Financial Document Analysis",
    keywords="rag, nlp, finance, 10-k, openai",
    url="https://github.com/yourusername/rag-pipeline",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "rag-pipeline=scripts.run_pipeline:main",
        ],
    },
)
