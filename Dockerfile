FROM python:3.8.11 AS builder
WORKDIR /app

#RUN pip install pandas\
#                gensim==3.8 \
#                konlpy \
#                python-dotenv \
#                elasticsearch \
#                textwrap3 \
#                xmltodict \
#                flask \
#                flask_restful \
#                flask_cors\
#                gunicorn\
#                nltk\
#                transformers\
#                networkx\
# RUN pip install -r requirements.txt
#COPY . .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.8.11 AS deployer
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /app /app
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn
COPY _ml_backend.py _ml_backend.py
COPY .env .env
COPY toshiba_nginx_ca.crt toshiba_nginx_ca.crt
COPY wsgi.py wsgi.py

ENV TZ=Asia/Seoul

ENTRYPOINT ["gunicorn --timeout=2000 --bind 0.0.0.0:5001 wsgi:app"]

# ENTRYPOINT ["bin/sh", "-c", "gunicorn", "--timeout=2000", "--bind", "0.0.0.0:5001", "wsgi:app"]
# ENTRYPOINT ["gunicorn", "--timeout" "2000", "--bind", "0.0.0.0:5001", "wsgi:app"]