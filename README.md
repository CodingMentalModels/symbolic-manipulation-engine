# Symbolic Manipulation Engine

Goal: An extensible engine for defining allowable symbolic manipulations, saving them into "Contexts" so that they can be reused / shared / etc., and creating tooling for parsing input (e.g. from a command line) and displaying output (LaTeX, etc.), such that doing math, physics, etc. is faster / easier than doing it on paper / blackboard / whiteboard, etc.

## User Stories

### Arithmetic

Given an arithmetic expression, like: $\ln(15) + 3 \cdot 2^2 - \sin(\tau/3)$, allow simplifications and computations of the result.  e.g.

$$
\begin{align}

\ln(15) + 3 \cdot 2^2 - \sin(\tau/3) = \\
\ln(15) + 12 - \sqrt{3}/2 = \\
13.84202...
\end{align}
$$


### Algebra

Given an algebraic expression like \$x^3 - 3x^2 + 3x - 1\$, factor and simplify it.

Manipulations with complex numbers, manipulations with Eisenstein Integers, manipulations with Clifford Algebras, etc.  Because each of these can be defined via a series of formal transformations on symbols, this should all be possible.

### Proofs

Given several hypothesized statements, manipulate them to reach a new conclusion.  E.g. given: $ x \in 2\mathbb{Z}, \enspace y \in 2\mathbb{Z} $, we should be able to show that $x + y \in 2\mathbb{Z}$.  

Something like:
$$
\begin{align}
\exists z \in \mathbb{Z} \enspace z = x + y \\
\exists a \in \mathbb{Z} \enspace 2a = x \\
\exists b \in \mathbb{Z} \enspace 2b = y \\
z = 2a + 2b \\
z = 2 \cdot (a + b) \\
\exists c \in \mathbb{Z} c = a + b
z \in 2\mathbb{Z}
\end{align}
$$

Many other similar cases, including:
- Set theory
- Propositional logic
- Calculus
- Physics

In each case, the point is to have a `Context` (or several) which provide valid transformations of symbols in that context.  


## Features

### Core Manipulations

- `Symbol`s are pure syntax; they have no inherent meaning but have `Type`s, format strings, and (optionally) `LaTeX` representations.
- `Type`s are also purely syntactic; they restrict the types of transformations that are valid.
- `Statement`s are trees of `Symbol`s
- `Transformation`s are rules for mapping one or more `Statement`s to a new `Statement`.  
- `Context`s are groups of saved, allowable `Transformation`s which can be imported and used, e.g. one might have a `Basic Algebra` `Context`, which can perform algebraic manipulations.

### Command Line Interface

In order to interact with the engine, we'll need an interface.  Let's start with a CLI, which allows:
- Defining new valid transformations
- Defining new contexts
- Importing contexts
- Stating statements
- Transforming statements

### Input Parsing

Inputting statements as trees is going to be painful, we want to be able to input them as plaintext and have them be parsed out.

### Database

- Stores and retreives `Context`s
- Stores and retreives saved work

### Pretty Printing

- `Symbol`s, `Statement`s, and `Transformations` can be given LaTeX representations, which can then be pretty printed.  

### Evaluation

It's going to be convenient to be able to evaluate arithmatic.  Create an extensible framework for allowing computation modules to run against certain `Type`s in certain `Context`s.  

### Further Ideas

- Embed this into a text editor, e.g. Sublime, VSCode, Neovim