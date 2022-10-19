# Time Log Analysis

Earlier this year I started keeping track of my time using an Android app called aTimeLogger by [Sergei Zaplitny](https://github.com/zaplitny):

http://www.atimelogger.com/

It keeps track of when you start and stop working on a task, and then you can export the data to a CSV file with two tables with an empty line separating them:

| Activity type | Duration | From | To | Comment|
| ------------- | -------- | ---- | -- | ------ |
| Sleep | 8:00 | 2022-01-01 22:00 | 2022-01-02 06:00 | |
| Eat | 1:00 | 2022-01-02 07:00 | 2022-01-02 08:00 | |
|||...
| Read | 1:00 | 2022-05-01 20:00 | 2022-05-01 21:00 | |

| Activity Type | Duration | % |
| ------------- | -------- | - |
| Sleep | 32:24 | 34.1 |
| Eat | 4:00 | 4.2 |
||...||
|Total | 94:24 | 100.0 |

This repo contains some scripts to read-in this CSV and generate a report. I ignore the second table and just use the first.

