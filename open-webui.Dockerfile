FROM ghcr.io/open-webui/open-webui:main

# Frontend static assets (served by SvelteKit)
COPY apps/admin/src/ras_admin/static/favicon.ico /app/build/static/favicon.ico
COPY apps/admin/src/ras_admin/static/favicon.ico /app/build/favicon.ico
COPY apps/admin/src/ras_admin/static/logo.png /app/build/static/favicon.png
COPY apps/admin/src/ras_admin/static/logo.png /app/build/static/favicon-96x96.png
COPY apps/admin/src/ras_admin/static/logo.png /app/build/favicon.png
COPY apps/admin/src/ras_admin/static/logo.png /app/build/static/logo.png
COPY apps/admin/src/ras_admin/static/logo.png /app/build/static/splash.png

# Backend static assets (served by FastAPI for API responses)
COPY apps/admin/src/ras_admin/static/logo.png /app/backend/open_webui/static/favicon.png
COPY apps/admin/src/ras_admin/static/logo.png /app/backend/open_webui/static/logo.png
COPY apps/admin/src/ras_admin/static/logo.png /app/backend/open_webui/static/splash.png
