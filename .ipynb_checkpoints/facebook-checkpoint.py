{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "phone = input(\"Phone: \")\n",
    "phone_digits = {\n",
    "    \"1\": \"one\",\n",
    "    \"2\": \"two\",\n",
    "    \"3\": \"three\",\n",
    "    \"4\": \"four\",\n",
    "    \"5\": \"five\",\n",
    "    \"6\": \"six\", \n",
    "    \"7\": \"seven\",\n",
    "    \"8\": \"eight\",\n",
    "    \"9\": \"nine\",\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "output = \"\"\n",
    "for ch in list(phone):\n",
    "    output += phone_digits.get(ch, \"\");\n",
    "printf(output);"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
