FROM node:24-slim AS build
WORKDIR /app
COPY web/package*.json ./
RUN npm ci --no-audit --no-fund
COPY features.json /features.json
COPY web ./
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
RUN node scripts/gen-features.mjs && npx next build

FROM node:24-slim
WORKDIR /app
RUN npm i -g serve@14
COPY --from=build /app/out ./out
EXPOSE 3000
CMD ["serve", "out", "-l", "3000", "-s"]
