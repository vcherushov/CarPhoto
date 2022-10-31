import psycopg2
from datetime import datetime

def table_insert(model):
    try:
        # Подключиться к существующей базе данных
        connection = psycopg2.connect(user="postgres",
                                      # пароль, который указали при установке PostgreSQL
                                      password="Naruto2010",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="car")

        cursor = connection.cursor()
        postgres_insert_query = """ INSERT INTO truck_detect (ID, MODEL, time)
                                           VALUES (%s,%s,%s)"""
        record_to_insert = (cursor.arraysize + 1, model, datetime.now())
        cursor.execute(postgres_insert_query, record_to_insert)

        connection.commit()
        count = cursor.rowcount
        print(count, "Запись успешно добавлена в таблицу mobile")

    except Exception as error:
        print("Ошибка при работе с PostgreSQL", error)
    finally:
        if connection:
            cursor.close()
            connection.close()
            print("Соединение с PostgreSQL закрыто")

table_insert('truck')