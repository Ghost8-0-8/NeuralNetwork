SCRIPTS = my_torch_generator my_torch_analyzer
TEMPLATE = cmd_template
GENERATOR_SCRIPT = ./GENERATOR/my_torch_generator.py
ANALYZER_SCRIPT = ./ANALYZER/my_torch_analyzer.py

all: $(SCRIPTS)

my_torch_generator: $(TEMPLATE)
	cp $(TEMPLATE) $@
	sed -i 's|PLACEHOLDER_SCRIPT|$(GENERATOR_SCRIPT)|g' $@
	chmod +x $@

my_torch_analyzer: $(TEMPLATE)
	cp $(TEMPLATE) $@
	sed -i 's|PLACEHOLDER_SCRIPT|$(ANALYZER_SCRIPT)|g' $@
	chmod +x $@

clean:
	rm -f $(SCRIPTS)

fclean: clean

re: fclean all

.PHONY: all clean fclean re
