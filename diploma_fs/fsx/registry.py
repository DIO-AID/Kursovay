"""Реестр реальных датасетов: ЕДИНСТВЕННОЕ место, где описан каждый датасет.

Его читают scripts/download_data.py (что и куда скачать) и scripts/run.py (--dataset имя).
Поля:
  uci_id    — ID в ucimlrepo (None, если файл кладётся вручную)
  file      — путь относительно data/ (diploma_fs/data или data/ в корне репозитория)
  target    — целевая колонка, ЯВНО (метаданные UCI бывают неверны: у 851 и 560 там не та цель)
  time_col  — столбец времени или None; если задан, разбиение по времени (только прошлое -> будущее)
  time_format — формат даты для pd.to_datetime (ОБЯЗАТЕЛЬНО для dd/mm: иначе 01/02 читается
              как 2 января и порядок ряда ломается молча); None — ISO/авто
  drop      — колонки-утечки и идентификаторы (удаляются до всего остального)
  cat_cols  — категориальные, даже если записаны числами (час, сезон, станция)
  sample    — сколько строк брать (None = все); для временных берётся непрерывный хвост ряда
  role      — main | classic | control (отрицательный контроль: ожидаем, что отбор не поможет)
  status    — draft (не проверен на скачанном файле) | verified
  --- только для временного прогноза (fsx/lags.py, scripts/run_forecast.py) ---
  horizon     — горизонт прогноза как интервал pandas ("1h", "1D"); признаки в момент t строятся
                только из данных до t − horizon
  known_ahead — колонки, известные заранее (расписание, календарь: день недели, NSM);
                они НЕ сдвигаются. Все остальные факторы в режиме «прогноз» берутся с лагом
  hour_col    — час в отдельной колонке (время = дата + час)
  --- только для читаемого отчёта (docs/REPORT_FORMAT.md) ---
  labels      — расшифровка колонок: {имя в файле: (смысл, единицы, роль)}. Роль — одно из:
                "цель", "известно заранее", "только прошлое", "удалено (и почему)", "время".
                Отчёт не должен показывать голые имена колонок.
  values      — расшифровка значений категорий: {колонка: {значение в файле: по-русски}}.
Правило отбора (зафиксировано до прогона, docs/DECISIONS.md): регрессия, >= 1000 строк,
5..40 числовых признаков, числовая цель задана явно, утечки удалены; spread — только описание.
"""

DATASETS = {
    "tetouan": dict(uci_id=849, file="849_tetouan.csv", target="Zone 1 Power Consumption",
                    time_col="DateTime", drop=["Zone 2  Power Consumption", "Zone 3  Power Consumption"],
                    cat_cols=[], sample=10000, role="main", status="draft",
                    horizon="1h", known_ahead=[],
                    labels={
                        "DateTime": ("Момент измерения", "", "время"),
                        "Temperature": ("Температура воздуха", "°C", "только прошлое"),
                        "Humidity": ("Относительная влажность", "%", "только прошлое"),
                        "Wind Speed": ("Скорость ветра", "м/с", "только прошлое"),
                        "general diffuse flows": ("Общая диффузная радиация", "Вт/м²", "только прошлое"),
                        "diffuse flows": ("Диффузная радиация", "Вт/м²", "только прошлое"),
                        "Zone 1 Power Consumption": ("Потребление зоны 1", "Вт", "цель"),
                        "Zone 2  Power Consumption": (
                            "Потребление зоны 2", "Вт",
                            "удалено: та же установка, почти копия зоны 1 — подсказала бы ответ"),
                        "Zone 3  Power Consumption": (
                            "Потребление зоны 3", "Вт",
                            "удалено: та же установка, почти копия зоны 1 — подсказала бы ответ"),
                    }),
    "steel": dict(uci_id=851, file="851_steel.csv", target="Usage_kWh", time_col="date",
                  time_format="%d/%m/%Y %H:%M", drop=["CO2(tCO2)"], cat_cols=["WeekStatus", "Day_of_week", "Load_Type"],
                  sample=10000, role="main", status="draft",
                  horizon="1h", known_ahead=["NSM", "WeekStatus", "Day_of_week"],
                  labels={
                      "date": ("Момент измерения", "", "время"),
                      "Usage_kWh": ("Потребление", "кВт·ч", "цель"),
                      "Lagging_Current_Reactive.Power_kVarh": (
                          "Реактивная энергия, отстающая", "кВАр·ч", "только прошлое"),
                      "Leading_Current_Reactive_Power_kVarh": (
                          "Реактивная энергия, опережающая", "кВАр·ч", "только прошлое"),
                      "Lagging_Current_Power_Factor": (
                          "Коэффициент мощности, отстающий", "%", "только прошлое"),
                      "Leading_Current_Power_Factor": (
                          "Коэффициент мощности, опережающий", "%", "только прошлое"),
                      "NSM": ("Секунд от полуночи", "с", "известно заранее"),
                      "WeekStatus": ("Рабочий день или выходной", "", "известно заранее"),
                      "Day_of_week": ("День недели", "", "известно заранее"),
                      "Load_Type": ("Тип нагрузки", "", "только прошлое"),
                      "CO2(tCO2)": (
                          "Выбросы CO2", "т",
                          "удалено: прямое следствие потребления в той же строке — утечка ответа"),
                  },
                  values={
                      "WeekStatus": {"Weekday": "рабочий день", "Weekend": "выходной"},
                      "Day_of_week": {"Monday": "понедельник", "Tuesday": "вторник",
                                      "Wednesday": "среда", "Thursday": "четверг",
                                      "Friday": "пятница", "Saturday": "суббота",
                                      "Sunday": "воскресенье"},
                      "Load_Type": {"Light_Load": "лёгкая нагрузка",
                                    "Medium_Load": "средняя нагрузка",
                                    "Maximum_Load": "максимальная нагрузка"},
                  }),
    "seoul_bike": dict(uci_id=560, file="560_seoul_bike.csv", target="Rented Bike Count",
                       time_col="Date", time_format="%d/%m/%Y", drop=[],
                       cat_cols=["Hour", "Seasons", "Holiday", "Functioning Day"],
                       sample=None, role="main", status="draft", hour_col="Hour",
                       horizon="1h", known_ahead=["Hour", "Seasons", "Holiday", "Functioning Day"]),
    "temp_forecast": dict(uci_id=514, file="514_temp_forecast.csv", target="Next_Tmax",
                          time_col="Date", time_format="%Y-%m-%d", drop=["Next_Tmin"], cat_cols=["station"],
                          sample=None, role="main", status="draft"),
    "gas_turbine": dict(uci_id=551, file="551_gas_turbine.csv", target="NOX", time_col=None,
                        drop=["CO", "Year"], cat_cols=[], sample=10000, role="main",
                        status="draft"),
    "auction": dict(uci_id=713, file="713_auction.csv", target="verification.time",
                    time_col=None, drop=["verification.result"], cat_cols=[],
                    sample=None, role="main", status="draft"),
    "thermography": dict(uci_id=925, file="925_thermography.csv", target="aveOralM",
                         time_col=None, drop=["aveOralF"], cat_cols=["Gender", "Age", "Ethnicity"],
                         sample=None, role="main", status="draft"),
    "lattice": dict(uci_id=1091, file="1091_lattice.csv", target="k-inf", time_col=None,
                    drop=["PPPF"], cat_cols=[], sample=5000, role="main", status="draft"),
    "concrete": dict(uci_id=165, file="165_concrete.csv",
                     target="Concrete compressive strength", time_col=None, drop=[],
                     cat_cols=[], sample=None, role="classic", status="draft"),
    "ccpp": dict(uci_id=294, file="294_ccpp.csv", target="PE", time_col=None, drop=[],
                 cat_cols=[], sample=None, role="classic", status="draft"),
    "wine_quality": dict(uci_id=186, file="186_wine_quality.csv", target="quality",
                         time_col=None, drop=[], cat_cols=["color"], sample=None,
                         role="control", status="draft"),
    "garment": dict(uci_id=597, file="597_garment.csv", target="actual_productivity",
                    time_col="date", time_format="%m/%d/%Y", drop=[], cat_cols=["quarter", "department", "day", "team"],
                    sample=None, role="control", status="draft"),
}
# Имена колонок в draft-записях взяты из карточек UCI и будут сверены с реально скачанными
# файлами (у Tetouan в названиях зон бывают двойные пробелы, у Gas Turbine — NOX/NOx).
