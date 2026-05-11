nithin
lang+ hindi

# --- NithinLang V1 Hindi Demo ---

# 1. Print
likho("Namaste, Duniya!")

# 2. Variables
a = 42
b = 8
likho("Jod:", a + b)

# 3. Conditional
agar a > 40:
    likho("a chaalisa se zyada hai")
warna:
    likho("a chaalisa se kam hai")

# 4. Loop
ke_liye i mein seema_mein(3):
    likho("Kadam:", i)

# 5. Function
kaam factorial(n):
    agar n <= 1:
        wapas 1
    wapas n * factorial(n - 1)

likho("5! =", factorial(5))

# 6. Math
likho("cos(0) =", math_cos(0))
likho("log(e) =", math_log(math_e))

# 7. JSON
obj = {"naam": "NithinLang", "version": 1}
json_str = nl_json_dump(obj)
likho("JSON:", json_str)

# 8. Hash
h = nl_hash("NithinLang", "sha256")
likho("SHA256:", h)

end nithin