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
COPY --from=build /app/dist /usr/share/nginx/html

# Backend base URL, substituted into the nginx config at startup.
ENV SD_API_URL=http://sd-api-backend:8000
EXPOSE 8080
