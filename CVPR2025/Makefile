# Makefile for compiling LaTeX files

# 文件名（不包含扩展名）
TARGET = main

# LaTeX编译器
LATEX = pdflatex
BIBTEX = bibtex

# 目标文件
PDF = $(TARGET).pdf

# 中间文件
TEMP_FILES = $(TARGET).aux $(TARGET).bbl $(TARGET).blg $(TARGET).log $(TARGET).out $(TARGET).toc

# 默认目标
all: $(PDF)

# 编译PDF
$(PDF): $(TARGET).tex
	$(LATEX) $(TARGET).tex
	$(BIBTEX) $(TARGET)
	$(LATEX) $(TARGET).tex
	$(LATEX) $(TARGET).tex

# 清理中间文件
clean:
	rm -f $(TEMP_FILES)

# 清理所有生成的文件
cleanall: clean
	rm -f $(PDF)

.PHONY: all clean cleanall
