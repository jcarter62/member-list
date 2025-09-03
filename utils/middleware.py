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

# class ClientIPLoggingMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         client_ip = request.client.host  # Extract client IP address
#         logging.info(f"Client IP: {client_ip} - {request.method} {request.url}")
#         response = await call_next(request)
#         return response

class ClientIPLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Get client IP from Cloudflare header or fallback
        client_ip = request.headers.get("cf-connecting-ip", request.client.host)
        # Extract username (example: from header or session)
        username = request.headers.get("cf-ray", "anonymous")
        platform = request.headers.get("sec-ch-ua-platform", "")
        if platform == "":
            platform = request.headers.get("user-agent", "unknown")

        logText = f"User: {username} | IP: {client_ip} | Platform: {platform} | {request.method} {request.url}"

        logging.info(logText)
        response = await call_next(request)
        return response


# Set up Jinja2Templates and inject company name as a global
templates_dir = os.path.join(os.getenv("APPFOLDER","~"), "templates")
company_name = os.getenv("COMPANY", "Default Company")
templates = Jinja2Templates(directory=templates_dir)
templates.env.globals["company"] = company_name

