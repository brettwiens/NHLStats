FROM python:3.7
EXPOSE 8501
WORKDIR /app
COPY Requirements.txt ./Requirements.txt
RUN pip3 install -r Requirements.txt
COPY . .
CMD streamlit run NHLStatistics.py