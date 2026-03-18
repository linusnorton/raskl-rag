FROM ghcr.io/open-webui/open-webui:main

COPY apps/admin/src/ras_admin/static/favicon.ico /app/build/static/favicon.ico
COPY apps/admin/src/ras_admin/static/favicon.ico /app/build/favicon.ico
COPY apps/admin/src/ras_admin/static/logo.png /app/build/static/favicon.png
COPY apps/admin/src/ras_admin/static/logo.png /app/build/favicon.png
COPY apps/admin/src/ras_admin/static/logo.png /app/build/static/logo.png
