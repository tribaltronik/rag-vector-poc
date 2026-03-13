# Makefile for RAG Vector PoC

.PHONY: help start up stop down destroy clean logs health

help:
	@echo "Available commands:"
	@echo "  make start    - Start all services (Docker Compose)"
	@echo "  make stop     - Stop all services"
	@echo "  make destroy  - Stop and remove all containers, volumes, and images"
	@echo "  make logs     - View logs from all services"
	@echo "  make health   - Check if all services are healthy"

start:
	docker-compose up -d
	@echo ""
	@echo "Services starting... Please wait 30 seconds."
	@echo "UI available at: http://localhost:8501"
	@echo "API available at: http://localhost:8000"
	@echo "Qdrant dashboard at: http://localhost:6333/dashboard"
	@echo ""

stop:
	docker-compose stop

down:
	docker-compose down

destroy:
	docker-compose down -v --remove-orphans
	@echo "All containers, volumes, and networks have been removed."

clean: destroy
	@echo "Use 'make destroy' to remove all data."

logs:
	docker-compose logs -f

health:
	@echo "Checking services..."
	@curl -s http://localhost:8000/health || echo "API not ready"
