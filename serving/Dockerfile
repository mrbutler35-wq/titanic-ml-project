FROM jupyter/base-notebook:x86_64-python-3.11

USER root

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /home/jovyan/work/models && \
    chown -R jovyan:users /home/jovyan/work

USER jovyan
WORKDIR /home/jovyan/work

COPY app.py .
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]