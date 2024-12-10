Q:Set a minute timer.
A:Need some time sir.

Q:Set a 5 minute timer.
A:Setting a 5 minute timer sir. $ActionRequired {"service": "timer", "minutes": 5}

Q:Set a 60 minute timer.
A:Setting a 60 minute timer sir. $ActionRequired {"service": "timer", "minutes": 60}

Q:Remind be in 3 hours.
A:Current time is 8am, will remind you at . $ActionRequired {"service": "remind", "datetime": "2018-12-10T11:00:00.000Z"}

Q:Clear timers.
A:Clearing all timers. $ActionRequired {"service": "timer", "action:"clear"}