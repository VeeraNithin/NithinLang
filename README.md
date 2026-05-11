🌍 What is NithinLang?

NithinLang is a revolutionary ecosystem built to break the English-language barrier in coding. It allows developers to write code in their native mother tongue (currently supporting Telugu, Hindi, and English), while executing at near C++ speeds using an LLVM JIT Compiler.

Built by the NGI Empire, NithinLang comes with "Batteries Included":

🧠 Zero-Cloud AI: Built-in AI functions (ai_adugu, ai_chudu) that run 100% locally. No API keys, no internet, total privacy. (Powered by N-Loader & Ollama).

🎮 2D Game Engine: Build hardware-accelerated games in 5 lines of code.

📊 Data Science Ready: Native ML wrappers for Pandas, Numpy, and Scikit-Learn.

⚡ LLVM Speed: Mathematical operations and tight loops are dynamically compiled to machine code for extreme performance.

🛠️ Installation

NithinLang is now officially available on PyPI for global access.

Step 1: Install NithinLang
Open your terminal and run:

pip install nithinlang


Step 2: Verify Installation

nithin version


Boom! The NGI Empire Core System is now active on your machine.

🎨 VS Code Syntax Highlighting

Write code in beautiful colors! NithinLang has an official VS Code extension.

Open VS Code.

Go to Extensions (Ctrl+Shift+X).

Search for NithinLang (Published by NGI-Empire) and click Install.

Open any .nl file and enjoy professional syntax highlighting for your native language!

💻 Quick Start & Syntax

You define the language at the top of your file using lang+.

Example 1: Telugu (Hello World & AI)

Create a file named main.nl:

nithin
lang+ telugu

# ==========================================
# 🔥 NithinLang Giga-Project: Telugu Core
# ==========================================

raayi("NithinLang loki Swagatham! (Welcome to NithinLang!)")

# Zero-Cloud AI integration in 1 line
ans = ai_adugu("What is the future of AI in India?")
raayi("AI Answer: ", ans)

end nithin


Example 2: Hindi (LLVM Fast Math)

Create a file named math.nl:

nithin
lang+ hindi

likho("NithinLang me aapka swagat hai!")

# Fast Math using LLVM JIT Compilation
a = math_sqrt(10000)
likho("10000 ka square root hai: ", a)

end nithin


To run your code, simply type:

nithin run main.nl


To benchmark hardware performance:

nithin bench main.nl


🔥 Command-Line Interface (CLI)

NithinLang comes with a powerful, rich-text CLI:

nithin new <project_name> : Scaffold a professional multi-lingual project.

nithin check <file.nl> : Static syntax and translation validation.

nithin ai-status : Check the status of your offline Neural Network (N-Loader).

nithin repl : Interactive native-language shell.

nithin langs : List all supported human languages.

🗺️ Roadmap (V2.0 & Beyond)

[x] Global PyPI Release

[x] VS Code Extension

[ ] Nithin Package Manager (NPM): Native package installation.

[ ] EXE/APK Builder: Compile your .nl files directly into .exe for Windows or .apk for Android.

[ ] Web Framework: Built-in fast web server for backend development.

[ ] More Languages: Adding Tamil, Kannada, Marathi, and Spanish.

🤝 Contributing

NithinLang is built for the community, by the community. Feel free to fork this repository, add your own native language JSON dictionary in src/nithinlang/dicts/, and submit a Pull Request!


BUILT WITH ❤️ BY NITHIN
CEO of NGI
A 17-year-old Diploma Student
