import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import os
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class ContextProcessorMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        # Set a context variable for the company name
        company = os.getenv("COMPANY", "Default Company")
        request.state.company = company

        response = await call_next(request)
        return response

class ClientIPLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host  # Extract client IP address
        logging.info(f"Client IP: {client_ip} - {request.method} {request.url}")
        response = await call_next(request)
        return response

# Set up Jinja2Templates and inject company name as a global
templates_dir = os.path.join(os.getenv("APPFOLDER","~"), "templates")
company_name = os.getenv("COMPANY", "Default Company")
templates = Jinja2Templates(directory=templates_dir)
templates.env.globals["company"] = company_name

