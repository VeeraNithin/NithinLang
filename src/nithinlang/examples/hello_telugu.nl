nithin
lang+ telugu

# --- NithinLang V1 Telugu Demo ---

# 1. Basic print
raayi("Namaskaram, Praapanicham!")

# 2. Variables & arithmetic
x = 10
y = 20
z = x + y
raayi("Yoogam:", z)

# 3. Conditional
ayithe z > 25:
    raayi("Z chaalaa peddi!")
lekapothe:
    raayi("Z chinna.")

# 4. While loop
count = 0
varaku count < 3:
    raayi("Count:", count)
    count = count + 1

# 5. For loop
ki_varaku i lo range_lo(5):
    raayi("Iterationi:", i)

# 6. Function definition
cheyyandi square(n):
    tiskonu n * n

result = square(7)
raayi("7 varga:", result)

# 7. File I/O
fh = f_open("test_out.txt", "w")
f_write(fh, "NithinLang Telugu test\n")
f_close(fh)

fh2     = f_open("test_out.txt", "r")
content = f_read(fh2)
f_close(fh2)
raayi("File content:", content)

# 8. Math
raayi("sqrt(144) =", math_sqrt(144))
raayi("pi =", math_pi)

# 9. ML (if pandas/sklearn available)
# df    = data_chudu("iris.csv")
# raayi(data_describe(df))

# 10. AI (if Ollama available)
# reply = ai_adugu("Oka chinna Telugu kavitha raayandi")
# raayi(reply)

end nithin