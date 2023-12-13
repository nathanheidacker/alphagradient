.. _algolib.template:

========
Template
========

.. currentmodule:: alphagradient.algolib

This is a template for documenting AlphaGradient algorithms. Feel free to copy
this document directly from `this link <https://raw.githubusercontent.com/nathanheidacker/alphagradient/blob/main/docs/source/algolib/TemplateAlgo.rst>`_, and follow its structure as a guide for creating your own documentation. Simply replace the content of each section according the guide written within, deleting unused sections or adding new ones as required. To begin, replace the topmost header with the name of your Algorithm. This header should follow the name of the class **exactly**, as should the name of this file. The reference at the top of the page should also follow the format :code:`.. _algolib.youralgorithmname:` in all lowercase.

Prerequisite Knowledge
----------------------
Before diving into the theory underlying the algorithm, use this section to briefly explain any obscure or otherwise advanced concepts utilized in your algorithm. Assume a working knowledge of basic finance and algorithmic trading concepts, but anything that requires more specialized knowledge should probably be explained here. Providing links to online articles explaining the concept (such as those on investopedia) is a good place to start. 

Essentially, we want to make sure that even someone with a relatively elementary understanding of finance has a good chance of understanding the algorithm. The level of understanding does not have to be thorough--rather, just enough that the results of the algorithm are explicable.

Algorithms that operate on relatively simple principles likely don't require this section. If your algorithm refrains from doing anything excessively complicated, or if such complexities are better explained in the 'theory' section, feel free to delete this section.

Theory
------
This is the meat and potatoes--what your algorithm does, how it works, and why it works. This section is essentially a much higher level, abstract version of your algorithm's 'cycle' docstring. Without getting into implementation details, explain what the algorithm tries to accomplish, how it accomplishes it, as well as why it's effective in the first place. 

This section will likely be (and likely *should* be) the longest of any contained within this document. For more experienced readers, it should leave very little to be questioned about the function of the algorithm. Novice readers should come away with some basic understanding of what the algorithm accomplishes.


Other Sections
--------------
It may be logical to compartmentalize your explanation into more discrete sections. We aren't too concerned with the organization of the documentation so long that it follows a logical flow from start to finish, and minimally contains the required sections (Theory and API). Subheadings (eg. splitting theory into many distinct subsections) is also perfectly acceptable, and often aids in readability. Feel free to add sections and organize structure to your liking. As long as it makes sense and maintains readability, its acceptable.


API
---
.. note::
    This section is purely for explanation of the sections that succeed it; please be sure to delete it before submission.

The sections below documenting the Algorithm's API should not contain any raw text. They exist purely to index an algorithms constructor, attributes, and methods, and should contain only autosummary directives with the respective Algorithm components.

Below, code examples of the necessary rst markup have been provided that produce the corresponding sections in HTML. Be sure to follow them exactly: direct the toctree to the relative path `../../generated` and use the `class.rst` template for the generator.

The API sections below are only those that are absolutely required. If your algorithm expands on these categories or adds new ones in a way that makes sense, feel free to change the structure appropriately.

Notes, warnings, deprecation notices and version notices are exceptions to the rule against raw text; please use them as you see fit to inform the user of necessary changes. 

Below is the rst markup necessary to produce the API sections.

.. code:: rst

    Constructor
    -----------
    .. autosummary::
        :toctree: ../generated
        :template: algorithm.rst

        TemplateAlgo

    Implementation
    --------------
    .. autosummary::
        :toctree: ../generated

        TemplateAlgo.setup
        TemplateAlgo.cycle
        TemplateAlgo.run

    Attributes
    ----------
    .. autosummary::
        :toctree: ../generated

        TemplateAlgo.template_property

    Internal Methods
    ----------------
    .. autosummary::
        :toctree: ../generated

        TemplateAlgo.template_method


Constructor
-----------
.. autosummary::
    :toctree: ../generated/
    :template: class.rst

    TemplateAlgo

Implementation
--------------
.. autosummary::
    :toctree: ../generated/

    TemplateAlgo.setup
    TemplateAlgo.cycle
    TemplateAlgo.run

Attributes
----------
.. autosummary::
    :toctree: ../generated/

    TemplateAlgo.template_property

Internal Methods
----------------
.. autosummary::
    :toctree: ../generated/

    TemplateAlgo.template_method