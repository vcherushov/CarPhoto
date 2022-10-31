import psycopg2

def update_table(car_id, number):
    try:
        # Подключиться к существующей базе данных
        connection = psycopg2.connect(user="postgres",
                                      # пароль, который указали при установке PostgreSQL
                                      password="Naruto2010",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="car")

        cursor = connection.cursor()
        print("Таблица до обновления записи")
        sql_select_query = """select * from truck_detect where id = %s"""
        cursor.execute(sql_select_query, (car_id,))
        record = cursor.fetchone()
        print(record)

        # Обновление отдельной записи
        sql_update_query = """Update truck_detect set number = %s where id = %s"""
        cursor.execute(sql_update_query, (number, car_id))
        connection.commit()
        count = cursor.rowcount
        print(count, "Запись успешно обновлена")

        print("Таблица после обновления записи")
        sql_select_query = """select * from truck_detect where id = %s"""
        cursor.execute(sql_select_query, (car_id,))
        record = cursor.fetchone()
        print(record)

    except Exception as error:
        print("Ошибка при работе с PostgreSQL", error)
    finally:
        if connection:
            cursor.close()
            connection.close()
            print("Соединение с PostgreSQL закрыто")

update_table(1, 'A123O')