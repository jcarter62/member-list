import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import os

class ContextProcessorMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        request.state.context = {
            "company": os.getenv('COMPANY', 'Company'),
        }
        response = await call_next(request)
        return response

class ClientIPLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host  # Extract client IP address
        logging.info(f"Client IP: {client_ip} - {request.method} {request.url}")
        response = await call_next(request)
        return response
