"""
FastAPI application for Westlands Water District employee directory.

This application models the legacy static employee listing and detail pages supplied
in the uploaded archive.  Data is stored in a SQLite database and exposed
through both an HTML user interface and a JSON API.  An administrative
interface allows a privileged user to create, update and delete employee
records.  New fields have been added to each employee: a mobile phone number
and a free‑form history text field.

To run the application locally use:

    uvicorn staff_app.main:app --reload

When the application starts for the first time it will automatically
create the SQLite database (``employees.db``) and seed it with the data from
``staff/data/employee-data.js`` found in the uploaded archive.  The seeding
process adds blank mobile and history fields to each imported record.
"""
from __future__ import annotations
import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import List, Optional
from fastapi import (FastAPI, Depends, HTTPException, Request, status)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from utils.middleware import ContextProcessorMiddleware, ClientIPLoggingMiddleware
from starlette.responses import FileResponse
import sqlite3
import base64
import hashlib
import hmac
import time
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware
from starlette.datastructures import UploadFile
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Database setup

load_dotenv()  # Load environment variables from .env file if present

COMPANY = os.getenv('COMPANY', 'Company')
DB_PATH = os.getenv('DBPATH', Path(__file__).resolve().parent / "employees.db")

templates_dir = os.path.join(os.getenv("APPFOLDER","~"), "templates")
templates = Jinja2Templates(directory=templates_dir)
# Expose company as a global in all templates
try:
    templates.env.globals["company"] = COMPANY
    templates.env.globals["image_subdir"] = os.getenv("IMAGE_SUBDIR", "images")
except Exception:
    pass

def get_db_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row factory configured."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_db():
    """FastAPI dependency providing a SQLite connection."""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


def dict_from_row(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dictionary."""
    d = dict(row)
    # start and end remain strings; convert to ISO for API output
    return d


# ---------------------------------------------------------------------------
# Validation helpers (email and US phone) and image upload helper

def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def normalize_us_phone(value: Optional[str]) -> Optional[str]:
    """Return phone formatted as (XXX) XXX-XXXX or empty string if blank.

    If the input does not contain a valid 10‑digit US number (optionally with
    a leading 1 country code), return None to indicate invalid.
    """
    if value is None:
        return ""
    digits = _digits_only(value)
    if not digits:
        return ""  # treat blank as empty string
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"


def is_valid_email(value: Optional[str]) -> bool:
    if value in (None, ""):
        return True  # optional
    # Simple but effective email pattern
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value) is not None


def save_uploaded_image(upload: Optional[UploadFile]) -> Optional[str]:
    """Save an uploaded image to the configured image directory and return the stored filename.

    - If IMAGE_ROOT is set: writes to IMAGE_ROOT/IMAGE_SUBDIR
    - Else: writes to static/IMAGE_SUBDIR
    Returns None if the upload is missing or invalid type.
    """
    if not upload or not getattr(upload, "filename", None):
        return None
    filename = os.path.basename(upload.filename)
    name, ext = os.path.splitext(filename)
    ext = ext.lower()
    allowed = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    if ext not in allowed:
        return None
    base = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._") or "image"
    final = f"{base}{ext}"

    image_root = os.getenv("IMAGE_ROOT")
    images_subdir = os.getenv("IMAGE_SUBDIR", "images")

    if image_root:
        images_dir = Path(image_root) / images_subdir
    else:
        images_dir = Path(__file__).resolve().parent / "static" / images_subdir
    images_dir.mkdir(parents=True, exist_ok=True)
    p = images_dir / final
    counter = 1
    while p.exists():
        final = f"{base}-{counter}{ext}"
        p = images_dir / final
        counter += 1
    # Write in chunks
    fileobj = getattr(upload, "file", None)
    if fileobj is None:
        return None
    with open(p, "wb") as out:
        while True:
            chunk = fileobj.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    try:
        fileobj.close()
    except Exception:
        pass
    return final


def parse_employee_js(js_path: Path) -> List[dict]:
    """Parse the legacy employee-data.js file into a list of dicts.

    The legacy file defines ``const employees = [ ... ];`` and may contain
    block and line comments.  This function extracts the JSON array and
    deserialises it into Python dictionaries.
    """
    text = js_path.read_text(encoding="utf-8")
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("Unable to locate JSON array in employee JS file")
    content = text[start : end + 1]
    # Remove block comments (/* ... */)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.S)
    # Remove line comments (// ... end of line)
    content = re.sub(r"//.*", "", content)
    # Now parse as JSON
    employees = json.loads(content)
    return employees


def seed_database_if_required() -> None:
    """Initialise the SQLite database with seed data from the legacy JS file.

    When the application starts it ensures the ``employees`` table exists.
    If the table is empty it populates it with the records parsed from
    ``employee-data.js``.  ``mobile`` and ``history`` are initialised to
    empty strings.
    """
    conn = get_db_connection()
    try:
        # Create table if it doesn't exist
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT UNIQUE,
                name TEXT NOT NULL,
                phone TEXT,
                mobile TEXT,
                position TEXT,
                email TEXT,
                url TEXT,
                dpt TEXT,
                img TEXT,
                history TEXT,
                start TEXT,
                end TEXT
            )
            """
        )
        # Check if there are existing rows
        cur = conn.execute("SELECT 1 FROM employees LIMIT 1")
        row = cur.fetchone()
        if row:
            return
        # Populate from JS
        js_path = Path(__file__).resolve().parent.parent / "unzipped_staff" / "staff" / "data" / "employee-data.js"
        if not js_path.exists():
            return
        employees = parse_employee_js(js_path)
        for emp in employees:
            conn.execute(
                """
                INSERT INTO employees (employee_id, name, phone, mobile, position, email, url, dpt, img, history, start, end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    emp.get("id"),
                    emp.get("name"),
                    emp.get("phone"),
                    "",  # mobile blank
                    emp.get("position"),
                    emp.get("email"),
                    emp.get("url"),
                    emp.get("dpt"),
                    emp.get("img"),
                    "",  # history blank
                    emp.get("start"),
                    emp.get("end"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Application setup
app = FastAPI(title="Westlands Water District Employees")

# Add middleware
app.add_middleware(ClientIPLoggingMiddleware)
app.add_middleware(ContextProcessorMiddleware)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", ""))

# ---------------------------------------------------------------------------
# Simple cookie‑based session helpers

SECRET_SESSION_KEY = os.environ.get("SECRET_KEY", "replace-with-random-key")

def create_session_token(username: str) -> str:
    """Return a signed session token for the given username."""
    timestamp = str(int(time.time()))
    data = f"{username}:{timestamp}"
    signature = hmac.new(SECRET_SESSION_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    token_raw = f"{data}:{signature}".encode()
    return base64.urlsafe_b64encode(token_raw).decode()


def verify_session_token(token: str) -> Optional[str]:
    """Return the username if token is valid, otherwise None."""
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        username, timestamp, signature = decoded.split(":")
        expected_signature = hmac.new(SECRET_SESSION_KEY.encode(), f"{username}:{timestamp}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            return None
        # Optionally enforce expiry (e.g., 1 day)
        # Here we allow tokens for 24 hours
        if time.time() - float(timestamp) > 86400:
            return None
        return username
    except Exception:
        return None

# Mount static files: CSS, images and JavaScript.  We copy assets from
# ``unzipped_staff/staff`` into our own ``static`` directory.  At runtime
# clients can retrieve these resources from ``/static/...``.
static_dir = Path(__file__).resolve().parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def on_startup() -> None:
    """Create the database and seed initial data on startup."""
    # When the server starts ensure the database file and table exist
    seed_database_if_required()


# ---------------------------------------------------------------------------
# Authentication helpers

ADMIN_USERNAME = os.environ.get("STAFF_APP_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("STAFF_APP_ADMIN_PASS", "password")


def is_authenticated(request: Request) -> bool:
    """Return True if the current session cookie corresponds to an admin user."""
    token = request.cookies.get("session")
    if not token:
        return False
    username = verify_session_token(token)
    return bool(username == ADMIN_USERNAME)


def require_login(request: Request) -> None:
    """Raise HTTPException if the current user is not authenticated."""
    if not is_authenticated(request):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")


# ---------------------------------------------------------------------------
# API endpoints

@app.get("/api/employees", response_model=List[dict])
def api_list_employees(db: sqlite3.Connection = Depends(get_db)):
    """Return a list of all employees as JSON."""
    cur = db.execute("SELECT * FROM employees ORDER BY name")
    rows = cur.fetchall()
    return [dict_from_row(row) for row in rows]


@app.get("/api/employees/{employee_id}", response_model=dict)
def api_get_employee(employee_id: int, db: sqlite3.Connection = Depends(get_db)):
    """Return details for a single employee or 404 if not found."""
    cur = db.execute("SELECT * FROM employees WHERE id = ?", (employee_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Employee not found")
    return dict_from_row(row)


@app.post("/api/employees", status_code=status.HTTP_201_CREATED)
async def api_create_employee(request: Request, db: sqlite3.Connection = Depends(get_db)):
    """Create a new employee.  Requires admin authentication.

    Accepts JSON payloads or URL‑encoded form data with keys matching the
    employee fields.  ``employee_id`` is ignored on input and assigned
    automatically.
    """
    require_login(request)
    data = {}
    # Try to parse JSON first
    try:
        data = await request.json()
    except Exception:
        # Fallback to form parsing
        form = await request.form()
        data = dict(form)
    # Extract values with defaults
    name = data.get("name")
    if not name:
        raise HTTPException(status_code=422, detail="Name is required")
    phone = data.get("phone")
    mobile = data.get("mobile")
    position = data.get("position")
    email = data.get("email")
    url_val = data.get("url")
    dpt = data.get("dpt")
    img = data.get("img")
    history = data.get("history")
    start = data.get("start")
    end = data.get("end")
    db.execute(
        """
        INSERT INTO employees (employee_id, name, phone, mobile, position, email, url, dpt, img, history, start, end)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (None, name, phone, mobile, position, email, url_val, dpt, img, history, start, end),
    )
    db.commit()
    cur = db.execute("SELECT * FROM employees ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    return dict_from_row(row)


@app.put("/api/employees/{employee_id}")
async def api_update_employee(employee_id: int, request: Request, db: sqlite3.Connection = Depends(get_db)):
    """Update an existing employee.  Requires admin authentication.

    Accepts JSON or URL‑encoded form data containing only the fields to update.
    """
    require_login(request)
    cur = db.execute("SELECT * FROM employees WHERE id = ?", (employee_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Employee not found")
    current = dict(row)
    # Parse incoming data
    data = {}
    try:
        data = await request.json()
    except Exception:
        form = await request.form()
        data = dict(form)
    updated = {
        "name": data.get("name", current["name"]),
        "phone": data.get("phone", current["phone"]),
        "mobile": data.get("mobile", current["mobile"]),
        "position": data.get("position", current["position"]),
        "email": data.get("email", current["email"]),
        "url": data.get("url", current["url"]),
        "dpt": data.get("dpt", current["dpt"]),
        "img": data.get("img", current["img"]),
        "history": data.get("history", current["history"]),
        "start": data.get("start", current["start"]),
        "end": data.get("end", current["end"]),
    }
    db.execute(
        """
        UPDATE employees
        SET name = ?, phone = ?, mobile = ?, position = ?, email = ?, url = ?, dpt = ?, img = ?, history = ?, start = ?, end = ?
        WHERE id = ?
        """,
        (
            updated["name"],
            updated["phone"],
            updated["mobile"],
            updated["position"],
            updated["email"],
            updated["url"],
            updated["dpt"],
            updated["img"],
            updated["history"],
            updated["start"],
            updated["end"],
            employee_id,
        ),
    )
    db.commit()
    cur = db.execute("SELECT * FROM employees WHERE id = ?", (employee_id,))
    row = cur.fetchone()
    return dict_from_row(row)


@app.delete("/api/employees/{employee_id}", status_code=status.HTTP_204_NO_CONTENT)
def api_delete_employee(employee_id: int, request: Request, db: sqlite3.Connection = Depends(get_db)):
    """Delete an employee.  Requires admin authentication."""
    require_login(request)
    cur = db.execute("SELECT 1 FROM employees WHERE id = ?", (employee_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Employee not found")
    db.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
    db.commit()
    return


# ---------------------------------------------------------------------------
# HTML views

@app.get("/", response_class=HTMLResponse)
def list_employees(request: Request, search: str = "", db: sqlite3.Connection = Depends(get_db)):
    """
    Render the employee listing.  Supports optional case‑insensitive search.

    Only employees whose start and end dates encompass today are shown.
    """
    cur = db.execute("SELECT * FROM employees")
    rows = cur.fetchall()
    today = _dt.datetime.utcnow().date()
    employees: List[sqlite3.Row] = []
    for row in rows:
        # Filter by date range
        start_str = row["start"]
        end_str = row["end"]
        try:
            start_date = _dt.datetime.fromisoformat(start_str.replace("Z", "+00:00")).date() if start_str else None
        except Exception:
            start_date = None
        try:
            end_date = _dt.datetime.fromisoformat(end_str.replace("Z", "+00:00")).date() if end_str else None
        except Exception:
            end_date = None
        if start_date and start_date > today:
            continue
        if end_date and end_date < today:
            continue
        # Filter by search term
        if search:
            term = search.lower()
            alltext = " ".join([
                (row["name"] or ""),
                (row["position"] or ""),
                (row["phone"] or ""),
                (row["mobile"] or ""),
                (row["email"] or ""),
                (row["employee_id"] or ""),
                (row["dpt"] or ""),
            ]).lower()
            if term not in alltext:
                continue
        employees.append(row)
    # Sort by name
    employees.sort(key=lambda r: r["name"] or "")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "employees": employees,
            "search": search,
            "logged_in": is_authenticated(request),
            "image_subdir": os.getenv("IMAGE_SUBDIR", "images"),
        },
    )


@app.get("/media/{filename}", response_class=HTMLResponse)
def media_file(filename: str, request: Request):
    """Serve media files from IMAGE_ROOT if configured, else from static/image_subdir."""
    image_root = os.getenv("IMAGE_ROOT")
    image_subdir = os.getenv("IMAGE_SUBDIR", "images")
    if image_root:
        fullpath = os.path.join(image_root, image_subdir, filename)
    else:
        fullpath = os.path.join(Path(__file__).resolve().parent, "static", image_subdir, filename)
    if os.path.exists(fullpath):
        media_type = "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg")) else "image/png"
        headers = {"Content-Type": media_type, "Cache-Control": "public, max-age=30"}
        return FileResponse(fullpath, media_type=media_type, headers=headers)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/employee/{employee_id}", response_class=HTMLResponse)
def employee_detail(employee_id: int, request: Request, db: sqlite3.Connection = Depends(get_db)):
    """Render the detail page for a single employee."""
    cur = db.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Employee not found")
    # Convert row to simple object-like structure for template ease
    class EmpObj:
        pass
    emp = EmpObj()
    for key in row.keys():
        setattr(emp, key, row[key])
    # Convert date strings to datetime objects for template compatibility
    from datetime import datetime
    if getattr(emp, "start", None):
        try:
            emp.start = datetime.strptime(emp.start[:10], "%Y-%m-%d")
        except Exception:
            emp.start = None
    if getattr(emp, "end", None):
        try:
            emp.end = datetime.strptime(emp.end[:10], "%Y-%m-%d")
        except Exception:
            emp.end = None
    return templates.TemplateResponse(
        "detail.html",
        {
            "request": request,
            "emp": emp,
            "logged_in": is_authenticated(request),
            "image_subdir": os.getenv("IMAGE_SUBDIR", "images"),
        },
    )


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    """Render the login form."""
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request):
    """Process the login form."""
    form = await request.form()
    username = form.get("username")
    password = form.get("password")
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        token = create_session_token(username)
        response = RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key="session",
            value=token,
            httponly=True,
            max_age=86400,
            path="/",
        )
        return response
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid username or password"},
    )


@app.get("/logout")
def logout(request: Request):
    """Log out the current user by clearing the session cookie."""
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="session", path="/")
    return response


@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(request: Request, db: sqlite3.Connection = Depends(get_db)):
    """Administrative dashboard listing all employees with edit/delete actions."""
    require_login(request)
    cur = db.execute("SELECT * FROM employees ORDER BY name")
    employees = cur.fetchall()
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "employees": employees,
        },
    )


@app.get("/admin/employee/new", response_class=HTMLResponse)
def new_employee_form(request: Request):
    """Render form for creating a new employee."""
    require_login(request)
    return templates.TemplateResponse("edit_employee.html", {
        "request": request,
        "employee": None,
        "form_action": "/admin/employee/new",
        "method": "POST",
    })


@app.post("/admin/employee/new", response_class=HTMLResponse)
async def create_employee(request: Request, db: sqlite3.Connection = Depends(get_db)):
    """Handle creation of a new employee via the admin form."""
    require_login(request)
    form = await request.form()
    # Extract file before casting to dict to preserve UploadFile instance
    upload = form.get("img_file")  # type: ignore[assignment]
    data = dict(form)
    name = data.get("name")
    if not name:
        return templates.TemplateResponse(
            "edit_employee.html",
            {
                "request": request,
                "employee": None,
                "form_action": "/admin/employee/new",
                "method": "POST",
                "error": "Name is required",
            },
        )
    # Uploaded image (if valid) overrides text field
    uploaded_filename = save_uploaded_image(upload if isinstance(upload, UploadFile) else None)
    effective_img = uploaded_filename or data.get("img")

    db.execute(
        """
        INSERT INTO employees (employee_id, name, phone, mobile, position, email, url, dpt, img, history, start, end)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("employee_id"),
            name,
            data.get("phone"),
            data.get("mobile"),
            data.get("position"),
            data.get("email"),
            data.get("url"),
            data.get("dpt"),
            effective_img,
            data.get("history"),
            data.get("start"),
            data.get("end"),
        ),
    )
    db.commit()
    return RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)


@app.get("/admin/employee/{employee_id}/edit", response_class=HTMLResponse)
def edit_employee_form(employee_id: int, request: Request, db: sqlite3.Connection = Depends(get_db)):
    """Render form for editing an existing employee."""
    require_login(request)
    cur = db.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Employee not found")
    # Convert row to object-like structure for Jinja
    class EmpObj:
        pass
    emp = EmpObj()
    for key in row.keys():
        setattr(emp, key, row[key])
    # Convert date strings to datetime objects for template compatibility
    from datetime import datetime
    if getattr(emp, "start", None):
        try:
            emp.start = datetime.strptime(emp.start[:10], "%Y-%m-%d")
        except Exception:
            emp.start = None
    if getattr(emp, "end", None):
        try:
            emp.end = datetime.strptime(emp.end[:10], "%Y-%m-%d")
        except Exception:
            emp.end = None
    return templates.TemplateResponse(
        "edit_employee.html",
        {
            "request": request,
            "employee": emp,
            "form_action": f"/admin/employee/{employee_id}/edit",
            "method": "POST",
        },
    )


@app.post("/admin/employee/{employee_id}/edit", response_class=HTMLResponse)
async def update_employee(
    employee_id: int,
    request: Request,
    db: sqlite3.Connection = Depends(get_db),
):
    """Handle updating an existing employee via the admin form."""
    require_login(request)
    # Fetch current record
    cur = db.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Employee not found")
    current = dict(row)
    form = await request.form()
    upload = form.get("img_file")  # potential UploadFile
    data = dict(form)
    name = data.get("name") or current["name"]
    # Normalize and validate phones
    raw_phone = data.get("phone") if data.get("phone") is not None else current.get("phone")
    raw_mobile = data.get("mobile") if data.get("mobile") is not None else current.get("mobile")
    phone_norm = normalize_us_phone(raw_phone)
    mobile_norm = normalize_us_phone(raw_mobile)

    if phone_norm is None:
        merged = {**current, **data}
        merged["phone"] = data.get("phone")
        merged["mobile"] = data.get("mobile")
        class EmpObj: pass
        emp = EmpObj(); [setattr(emp, k, v) for k, v in merged.items()]
        return templates.TemplateResponse(
            "edit_employee.html",
            {
                "request": request,
                "employee": emp,
                "form_action": f"/admin/employee/{employee_id}/edit",
                "method": "POST",
                "error": "Phone must be a valid US number (10 digits).",
            },
        )

    if mobile_norm is None:
        merged = {**current, **data}
        merged["phone"] = data.get("phone")
        merged["mobile"] = data.get("mobile")
        class EmpObj: pass
        emp = EmpObj(); [setattr(emp, k, v) for k, v in merged.items()]
        return templates.TemplateResponse(
            "edit_employee.html",
            {
                "request": request,
                "employee": emp,
                "form_action": f"/admin/employee/{employee_id}/edit",
                "method": "POST",
                "error": "Mobile must be a valid US number (10 digits).",
            },
        )

    email = data.get("email", current["email"]) or ""
    if not is_valid_email(email):
        merged = {**current, **data}
        merged["phone"] = data.get("phone")
        merged["mobile"] = data.get("mobile")
        class EmpObj: pass
        emp = EmpObj(); [setattr(emp, k, v) for k, v in merged.items()]
        return templates.TemplateResponse(
            "edit_employee.html",
            {
                "request": request,
                "employee": emp,
                "form_action": f"/admin/employee/{employee_id}/edit",
                "method": "POST",
                "error": "Email must be a valid email address.",
            },
        )

    position = data.get("position", current["position"]).strip() if data.get("position") is not None else current["position"]
    url_val = data.get("url", current["url"]) or None
    dpt = data.get("dpt", current["dpt"]) or None
    # Uploaded image (if valid) overrides text field; else keep existing when both blank
    uploaded_filename = save_uploaded_image(upload if isinstance(upload, UploadFile) else None)
    if uploaded_filename:
        effective_img = uploaded_filename
    else:
        effective_img = data.get("img") if data.get("img") else current.get("img")
    img = effective_img or None

    history = data.get("history", current["history"]) or None
    start = data.get("start", current["start"]) or None
    end = data.get("end", current["end"]) or None

    db.execute(
        """
        UPDATE employees
        SET name = ?, phone = ?, mobile = ?, position = ?, email = ?, url = ?, dpt = ?, img = ?, history = ?, start = ?, end = ?
        WHERE employee_id = ?
        """,
        (
            name,
            phone_norm,
            mobile_norm,
            position,
            email or None,
            url_val,
            dpt,
            img,
            history,
            start,
            end,
            employee_id,
        ),
    )
    db.commit()
    return RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)


@app.post("/admin/employee/{employee_id}/delete")
def delete_employee(
    employee_id: int, request: Request, db: sqlite3.Connection = Depends(get_db)
):
    """Delete an employee from the admin dashboard."""
    require_login(request)
    # Check if exists
    cur = db.execute("SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,))
    row = cur.fetchone()
    if row:
        db.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))
        db.commit()
    return RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)


if __name__ == "__main__":
    pass
