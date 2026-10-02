"""Tầng phục vụ — FastAPI + PostgreSQL (phương án B, docs/03 §3.2).

Ba tầng, phụ thuộc một chiều: ``routes`` → ``services`` → truy cập dữ liệu (``db``,
``models_orm``). Logic mô hình không nằm ở đây mà ở ``src/`` dùng chung với notebook.
"""
