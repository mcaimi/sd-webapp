# syntax=docker/dockerfile:1

# ---- build: compile the SPA
FROM node:24-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# ---- runtime: unprivileged nginx (port 8080, non-root, works with OpenShift's arbitrary UIDs)
FROM nginxinc/nginx-unprivileged:stable-alpine
COPY nginx/security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --chmod=755 nginx/15-auth-token.envsh /docker-entrypoint.d/15-auth-token.envsh
COPY --from=build /app/dist /usr/share/nginx/html

# Backend base URL, substituted into the nginx config at startup.
ENV SD_API_URL=http://sd-api-backend:8000
# Access token for the UI and /api (SD_WEB_TOKEN). When unset, one is generated
# at startup and printed to the container log.
EXPOSE 8080
