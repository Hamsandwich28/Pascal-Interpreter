from enum import Enum


class ErrorCode(Enum):
    UNEXPECTED_TOKEN    = 'Unexpected token'
    ID_NOT_FOUND        = 'Identifier not found'
    DUPLICATE_ID        = 'Duplicate id found'


class Error(Exception):
    def __init__(self, error_code=None, token=None, message=None):
        self.error_code = error_code
        self.token = token
        self.message = f'{self.__class__.__name__}: {message}'

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.__str__()


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class DefineError(Error):
    pass


#   ===================
#   ТИПЫ ТОКЕНОВ
#   ===================


class TokenType(Enum):
    PLUS            = '+'
    MINUS           = '-'
    MUL             = '*'
    FLOAT_DIV       = '/'
    LPAREN          = '('
    RPAREN          = ')'
    SEMI            = ';'
    DOT             = '.'
    COLON           = ':'
    COMMA           = ','
    EQ              = '='
    LW              = '<'
    GR              = '>'
    # RESERVED
    PROGRAM         = 'PROGRAM'
    INTEGER         = 'INTEGER'
    REAL            = 'REAL'
    INTEGER_DIV     = 'DIV'
    VAR             = 'VAR'
    IF              = 'IF'
    THEN            = 'THEN'
    ELSE            = 'ELSE'
    BEGIN           = 'BEGIN'
    END             = 'END'
    # DINAMIC
    ID              = 'ID'
    INTEGER_CONST   = 'INTEGER_CONST'
    REAL_CONST      = 'REAL_CONST'
    ASSIGN          = ':='
    LE              = '<='
    GE              = '>='
    EOF             = 'EOF'


#   ===================
#   ТОКЕНЫ
#   ===================


class Token(object):
    def __init__(self, token_type, value, lineno=None, column=None):
        self.type = token_type
        self.value = value
        self.lineno = lineno,
        self.column = column

    def __str__(self):
        return 'Token({type}, {value}, position={lineno}:{column})'.format(
            type=self.type,
            value=repr(self.value),
            lineno=self.lineno[-1],
            column=self.column
        )

    def __repr__(self):
        return self.__str__()


def _build_reserved_keywords():
    tt_list = list(TokenType)
    start_index = tt_list.index(TokenType.PROGRAM)
    end_index = tt_list.index(TokenType.END)
    reserved_keywords = {
        token_type.value: token_type
        for token_type in tt_list[start_index:end_index + 1]
    }
    return reserved_keywords


RESERVED_KEYWORDS = _build_reserved_keywords()


#   ===================
#   ЛЕКСИЧЕСКИЙ АНАЛИЗАТОР
#   ===================


class Lexer(object):
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]
        self.lineno = 1
        self.column = 1

    def error(self):
        msg = f'Lexer error on {self.current_char} ' + \
            f'line: {self.lineno} column: {self.column}'
        raise LexerError(message=msg)

    def advance(self):
        if self.current_char == '\n':
            self.lineno += 1
            self.column = 0

        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]
            self.column += 1

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        while self.current_char != '}':
            self.advance()
        self.advance()

    def number(self):
        token = Token(token_type=None, value=None,
                      lineno=self.lineno, column=self.column)

        # БИНАРНЫЙ ВВОД
        result = ''
        while (
            self.current_char is not None and
            self.current_char == '0' or self.current_char == '1'
        ):
            result += self.current_char
            self.advance()

        if self.current_char in list(str(i) for i in range(2, 10)):
            self.error()

        token.type = TokenType.INTEGER_CONST
        token.value = int(result, 2)

        return token

    def _id(self):
        token = Token(token_type=None, value=None,
                      lineno=self.lineno, column=self.column)

        value = ''
        while self.current_char is not None and self.current_char.isalnum():
            value += self.current_char
            self.advance()

        token_type = RESERVED_KEYWORDS.get(value.upper())
        if token_type is None:
            token.type = TokenType.ID
            token.value = value
        else:
            token.type = token_type
            token.value = value.upper()

        return token

    def get_next_token(self):
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '{':
                self.advance()
                self.skip_comment()
                continue

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == ':' and self.peek() == '=':
                token = Token(
                    token_type=TokenType.ASSIGN,
                    value=TokenType.ASSIGN.value,
                    lineno=self.lineno,
                    column=self.column
                )
                self.advance()
                self.advance()
                return token

            if self.current_char == '<' and self.peek() == '=':
                token = Token(
                    token_type=TokenType.LE,
                    value=TokenType.LE.value,
                    lineno=self.lineno,
                    column=self.column
                )
                self.advance()
                self.advance()
                return token

            if self.current_char == '>' and self.peek() == '=':
                token = Token(
                    token_type=TokenType.GE,
                    value=TokenType.GE.value,
                    lineno=self.lineno,
                    column=self.column
                )
                self.advance()
                self.advance()
                return token

            try:
                token_type = TokenType(self.current_char)
            except ValueError:
                self.error()
            else:
                token = Token(
                    token_type=token_type,
                    value=token_type.value,
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                return token

        return Token(token_type=TokenType.EOF, value=None,
                     lineno=self.lineno, column=self.column)


#   ===================
#   НОДЫ
#   ===================


class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class Compound(AST):
    def __init__(self):
        self.children = []


class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class ConditionStatement(AST):
    def __init__(self, condition, if_body, else_body):
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body


class Condition(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Var(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class NoOp(AST):
    pass


class Program(AST):
    def __init__(self, name, block):
        self.name = name
        self.block = block


class Block(AST):
    def __init__(self, declarations, compound_statement):
        self.declarations = declarations
        self.compound_statement = compound_statement


class VarDecl(AST):
    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node


class Type(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


#   ===================
#   ПАРСЕР
#   ===================


class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.get_next_token()

    def get_next_token(self):
        return self.lexer.get_next_token()

    def error(self, error_code, token):
        raise ParserError(
            error_code=error_code,
            token=token,
            message=f'{error_code.value} -> {token}'
        )

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(
                error_code=ErrorCode.UNEXPECTED_TOKEN,
                token=self.current_token
            )

    def program(self):
        self.eat(TokenType.PROGRAM)
        var_node = self.variable()
        prog_name = var_node.value
        self.eat(TokenType.SEMI)
        block_node = self.block()
        program_node = Program(prog_name, block_node)
        self.eat(TokenType.DOT)
        return program_node

    def block(self):
        declaration_nodes = self.declarations()
        compound_statement_node = self.compound_statement()
        node = Block(declaration_nodes, compound_statement_node)
        return node

    def declarations(self):
        declarations = []
        if self.current_token.type == TokenType.VAR:
            self.eat(TokenType.VAR)
            while self.current_token.type == TokenType.ID:
                var_decl = self.variable_declaration()
                declarations.extend(var_decl)
                self.eat(TokenType.SEMI)

        return declarations

    def variable_declaration(self):
        var_nodes = [Var(self.current_token)]
        self.eat(TokenType.ID)

        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            var_nodes.append(Var(self.current_token))
            self.eat(TokenType.ID)

        self.eat(TokenType.COLON)

        type_node = self.type_spec()
        var_declarations = [
            VarDecl(var_node, type_node)
            for var_node in var_nodes
        ]
        return var_declarations

    def type_spec(self):
        token = self.current_token
        if self.current_token.type == TokenType.INTEGER:
            self.eat(TokenType.INTEGER)
        else:
            self.eat(TokenType.REAL)
        node = Type(token)
        return node

    def compound_statement(self):
        self.eat(TokenType.BEGIN)
        nodes = self.statement_list()
        self.eat(TokenType.END)

        root = Compound()
        for node in nodes:
            root.children.append(node)

        return root

    def statement_list(self):
        node = self.statement()

        results = [node]

        while self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            results.append(self.statement())

        return results

    def statement(self):
        if self.current_token.type == TokenType.BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == TokenType.IF:
            node = self.condition_statement()
        elif self.current_token.type == TokenType.ID:
            node = self.assignment_statement()
        else:
            node = self.empty()
        return node

    def assignment_statement(self):
        left = self.variable()
        token = self.current_token
        self.eat(TokenType.ASSIGN)
        right = self.expr()
        node = Assign(left, token, right)
        return node

    def condition_statement(self):
        self.eat(TokenType.IF)
        condition = self.condition()
        self.eat(TokenType.THEN)
        if_body = self.statement()
        self.eat(TokenType.ELSE)
        else_body = self.statement()
        node = ConditionStatement(
            condition=condition, if_body=if_body, else_body=else_body
        )
        return node

    def condition(self):
        left = self.expr()
        if (
                self.current_token.type in
                (TokenType.LW, TokenType.LE, TokenType.GR,
                 TokenType.GE, TokenType.EQ)
        ):
            token = self.current_token
            if token.type == TokenType.LW:
                self.eat(TokenType.LW)
            elif token.type == TokenType.LE:
                self.eat(TokenType.LE)
            elif token.type == TokenType.GR:
                self.eat(TokenType.GR)
            elif token.type == TokenType.GE:
                self.eat(TokenType.GE)
            elif token.type == TokenType.EQ:
                self.eat(TokenType.EQ)

            node = Condition(left=left, op=token, right=self.expr())
        else:
            self.error(
                error_code=ErrorCode.UNEXPECTED_TOKEN,
                token=self.current_token
            )
        return node

    def variable(self):
        node = Var(self.current_token)
        self.eat(TokenType.ID)
        return node

    def empty(self):
        return NoOp()

    def expr(self):
        node = self.term()

        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            token = self.current_token
            if token.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
            elif token.type == TokenType.MINUS:
                self.eat(TokenType.MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def term(self):
        node = self.factor()

        while (
                self.current_token.type in
                (TokenType.MUL, TokenType.INTEGER_DIV, TokenType.FLOAT_DIV)
        ):
            token = self.current_token
            if token.type == TokenType.MUL:
                self.eat(TokenType.MUL)
            elif token.type == TokenType.INTEGER_DIV:
                self.eat(TokenType.INTEGER_DIV)
            elif token.type == TokenType.FLOAT_DIV:
                self.eat(TokenType.FLOAT_DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def factor(self):
        token = self.current_token
        if token.type == TokenType.PLUS:
            self.eat(TokenType.PLUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == TokenType.MINUS:
            self.eat(TokenType.MINUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == TokenType.INTEGER_CONST:
            self.eat(TokenType.INTEGER_CONST)
            return Num(token)
        elif token.type == TokenType.REAL_CONST:
            self.eat(TokenType.REAL_CONST)
            return Num(token)
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
        else:
            node = self.variable()
            return node

    def parse(self):
        """
        program : PROGRAM variable SEMI block DOT
        block : declarations compound_statement
        declarations : (VAR (variable_declaration SEMI)+)? procedure_declaration*
        variable_declaration : ID (COMMA ID)* COLON type_spec
        compound_statement : BEGIN statement_list END
        statement_list : statement
                       | statement SEMI statement_list
        statement : compound_statement
                  | assignment_statement
                  | condition_statement
                  | empty
        assignment_statement : variable ASSIGN expr
        condition_statement : IF condition THEN statement_list ELSE statement_list
        condition : expr (LW | LE | GR | GE | EQ) expr
        empty :
        expr : term ((PLUS | MINUS) term)*
        term : factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*
        factor : PLUS factor
               | MINUS factor
               | INTEGER_CONST
               | REAL_CONST
               | LPAREN expr RPAREN
               | variable
        variable: ID
        """
        node = self.program()
        if self.current_token.type != TokenType.EOF:
            self.error(error_code=ErrorCode.UNEXPECTED_TOKEN,
                       token=self.current_token)

        return node


#   ===================
#   ЧТЕНИЕ НОД
#   ===================

class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f'No visit_{type(node).__name__} method')


#   ===================
#   ПЕРЕМЕННЫЕ
#   ===================


class Symbol(object):
    def __init__(self, name, type=None):
        self.name = name
        self.type = type


class VarSymbol(Symbol):
    def __init__(self, name, type):
        super().__init__(name, type)

    def __str__(self):
        return "<{class_name}(name='{name}', type='{type}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,
            type=self.type
        )

    __repr__ = __str__


class BuiltinTypeSymbol(Symbol):
    def __init__(self, name):
        super().__init__(name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<{class_name}(name='{name}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,
        )


class SymbolTable(object):
    def __init__(self):
        self._symbols = {}
        self._init_builtins()

    def _init_builtins(self):
        self.define(BuiltinTypeSymbol('INTEGER'))
        self.define(BuiltinTypeSymbol('REAL'))

    def __str__(self):
        s = 'Symbols: {symbols}'.format(
            symbols=[value for value in self._symbols.values()]
        )
        return s

    __repr__ = __str__

    def define(self, symbol):
        # print('Define: %s' % symbol)
        self._symbols[symbol.name] = symbol

    def lookup(self, name):
        # print('Lookup: %s' % name)
        symbol = self._symbols.get(name)
        return symbol


class SymbolTableBuilder(NodeVisitor):
    def __init__(self):
        self.symtab = SymbolTable()

    def error(self, error_code, token):
        raise DefineError(
            error_code=error_code,
            token=token,
            message=f'{error_code.value} -> {token}'
        )

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_Program(self, node):
        self.visit(node.block)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visit_Num(self, node):
        pass

    def visit_UnaryOp(self, node):
        self.visit(node.expr)

    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)

    def visit_NoOp(self, node):
        pass

    def visit_VarDecl(self, node):
        type_name = node.type_node.value
        type_symbol = self.symtab.lookup(type_name)
        var_name = node.var_node.value
        var_symbol = VarSymbol(var_name, type_symbol)
        if self.symtab.lookup(var_name) is not None:
            self.error(
                error_code=ErrorCode.DUPLICATE_ID,
                token=node.var_node.token
            )

        self.symtab.define(var_symbol)

    def visit_ConditionStatement(self, node):
        self.visit(node.condition)
        self.visit(node.if_body)
        self.visit(node.else_body)

    def visit_Condition(self, node):
        pass

    def visit_Assign(self, node):
        var_name = node.left.value
        var_symbol = self.symtab.lookup(var_name)
        if var_symbol is None:
            self.error(error_code=ErrorCode.ID_NOT_FOUND, token=node.left.value)

        self.visit(node.right)

    def visit_Var(self, node):
        var_name = node.value
        var_symbol = self.symtab.lookup(var_name)

        if var_symbol is None:
            raise NameError(repr(var_name))


#   ===================
#   ИНТЕРПРЕТАТОР
#   ===================


class Interpreter(NodeVisitor):
    def __init__(self, tree):
        self.tree = tree
        self.GLOBAL_MEMORY = {}

    def visit_Program(self, node):
        self.visit(node.block)

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_VarDecl(self, node):
        pass

    def visit_Type(self, node):
        pass

    def visit_BinOp(self, node):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == TokenType.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == TokenType.MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == TokenType.INTEGER_DIV:
            return self.visit(node.left) // self.visit(node.right)
        elif node.op.type == TokenType.FLOAT_DIV:
            return float(self.visit(node.left)) / float(self.visit(node.right))

    def visit_Num(self, node):
        return node.value

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == TokenType.PLUS:
            return +self.visit(node.expr)
        elif op == TokenType.MINUS:
            return -self.visit(node.expr)

    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)

    def visit_ConditionStatement(self, node):
        condition = self.visit(node.condition)
        print(f'\n{condition}\n')
        if condition:
            self.visit(node.if_body)
        else:
            self.visit(node.else_body)

    def visit_Condition(self, node):
        exits = {
            '<': lambda x, y: x < y,
            '>': lambda x, y: x > y,
            '<=': lambda x, y: x <= y,
            '>=': lambda x, y: x >= y,
            '=': lambda x, y: x == y
        }
        return exits[node.token.value](
            self.visit(node.left), self.visit(node.right)
        )

    def visit_Assign(self, node):
        var_name = node.left.value
        var_value = self.visit(node.right)
        self.GLOBAL_MEMORY[var_name] = var_value

    def visit_Var(self, node):
        var_name = node.value
        var_value = self.GLOBAL_MEMORY.get(var_name)
        return var_value

    def visit_NoOp(self, node):
        pass

    def interpret(self):
        tree = self.tree
        if tree is None:
            return ''
        return self.visit(tree)


def main():
    import sys
    text = open(sys.argv[1], 'r').read()

    lexer = Lexer(text)
    parser = Parser(lexer)
    tree = parser.parse()
    symtab_builder = SymbolTableBuilder()
    symtab_builder.visit(tree)

    print('\nТАБЛИЦА СИМВОЛОВ:')
    print(symtab_builder.symtab)

    interpreter = Interpreter(tree)
    result = interpreter.interpret()

    print('\nГЛОБАЛЬНАЯ ПАМЯТЬ:')
    for key, val in sorted(interpreter.GLOBAL_MEMORY.items()):
        print(f'{key} = {val}')


if __name__ == '__main__':
    main()
