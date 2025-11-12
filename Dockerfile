# Python 3.12 Lambda base
FROM public.ecr.aws/lambda/python:3.12

# Install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy handler
COPY src/ ${LAMBDA_TASK_ROOT}

# Set the handler function (module.function)
CMD ["handler.lambda_handler"]
