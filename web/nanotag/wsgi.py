"""WSGI entry point for gunicorn:  gunicorn 'nanotag.wsgi:app'"""
from .web import create_app

app = create_app()
