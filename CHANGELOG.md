1.1.0 (2024-10-08)
------------------

New
~~~
- License, version badge. [Anthony Mahanna]

Fix
~~~
- Update version. [Anthony Mahanna]
- Comment. [Anthony Mahanna]
- Readme image link. [Anthony Mahanna]
- `index.rst` [Anthony Mahanna]
- Typo (again) [Anthony Mahanna]
- Typo. [Anthony Mahanna]
- Notebook alg. [Anthony Mahanna]

Other
~~~~~
- Smart graph support (#61) [Anthony Mahanna]

  * smart graph support | initial commit

  * cleanup: `_create_node_attr_dict`

  * new: `cast_to_string`

  * cleanup: overrides

  * fix: lint

  * fix: `overwrite_graph`

  * new: `GraphNotEmpty` exception

  * lock deps

  * remove: `_get_smart_id`

  * new: `test_load_graph_from_nxadb_as_smart_graph`

  * new: `add_nodes_from_override`

  * fix: typo

  * fix: lint

  * fix: pyproject

  * add comment

  * `overwrite_graph` docstring

  * update `adbnx-adapter` version

  * fix: var name

  * fix: `GraphNotEmpty` logic

  * fix: whitespace

  * fix: drop instead of truncate

  * Revert "fix: drop instead of truncate"

  This reverts commit 11347c91d521a246f4e1a5694d278f6d32137d8b.

  * add `overwrite_graph` coverage

  * fix: drop graph instead of truncate

  * fix: docstring

  * fix: `name` docstring
- Merge branch 'main' of https://github.com/arangodb/nx-arangodb.
  [Anthony Mahanna]
- !gitchangelog (#58) [aMahanna, github-actions[bot]]


1.0.1 (2024-09-03)
------------------
- Bump: version. [Anthony Mahanna]
- Doc fixes (#57) [Anthony Mahanna]

  * fix: badges

  * attempt: fix doc config

  * remove: dep

  * attempt fix: `currentmodule`

  * attempt: update conf

  * remove: mock

  * cleanup

  * update readme

  * update readme

  * update: fail on warning

  * attempt: remove `nx-arangodb`

  * revert cf20f43

  * fix eof


1.0.0 (2024-09-03)
------------------

New
~~~
- Code-ql (#56) [Anthony Mahanna]
- Ipynb notebook (#53) [Anthony Mahanna]

  * new: notebook

  * fix readme
- ISSUE_TEMPLATE (#51) [Anthony Mahanna]
- Rtd file. [Anthony Mahanna]
- Github Actions CI (#49) [Anthony Mahanna]

  * new: actions ci

  * new: changelog file

  * new: sphinx template

  * temp: move to `disabled-workflows`

  * cleanup

  * cleanup

  * more cleanup

  * 0.1?

  * 1.0
- `langchain` plugin (#44) [Anthony Mahanna]
- `use_experimental_views`  (#41) [Anthony Mahanna]

  * new: `use_experimental_views`

  * set experimental views

  * attempt fix: constructor

  * set experimental again
- `copy` method. [Anthony Mahanna]
- `dict` directory (#27) [Anthony Mahanna]

  * new: `dict` directory

  * cleanup
- Add `phenolrs` wheel (#6) [Anthony Mahanna]
- Enable `black`, `isort`, `flake8`, `mypy` (#5) [Anthony Mahanna]

  * new: enable `black`, `isort`, `flake8`

  * checkpoint: `mypy`

  * cleanup: `build.yaml`

  * fix: `build.yaml`

  * fix: `uses`

  * add `type: ignore`

  * fix: flake

  * fix: `phenolrs`

  * fix: mypy

  * restructure class order in `dict.py`

  * bring back `__repr__` and `__str__`

  * cleanup

  * rename: `_to_nx_graph`
- Invoke `adbnx_adapter` from `nxadb.Graph` constructor (#4) [Anthony
  Mahanna]

  * new: invoke `adbnx_adapter` from `nxadb.Graph` constructor

  * fix: conditional

  * fix: delete graph after creation

  * update graph_loader defaults

  * cleanup: test

  * cleanup
- Colab link. [Anthony Mahanna]
- `run_on_gpu` dev param. [Anthony Mahanna]
- Optional `gpu` dependency. [Anthony Mahanna]
- `to_networkx_class` [Anthony Mahanna]
- `test_bc` [Anthony Mahanna]
- Readme. [Anthony Mahanna]

Fix
~~~
- Readme link. [Anthony Mahanna]
- Limit gpu tests (#46) [Anthony Mahanna]
- `requires-python` [Anthony Mahanna]
- Shortest path `source` & `target` [Anthony Mahanna]
- Cache `nxcg` graph instead of coo representation (#31) [Anthony
  Mahanna]

  * fix: cache `nxcg` graph instead of coo representation

  * fix lint

  * fix print statements
- Typo. [Anthony Mahanna]
- Param name. [Anthony Mahanna]
- `run_on_gpu` dev param. [Anthony Mahanna]
- Centrality import. [Anthony Mahanna]
- Graph subclassing. [Anthony Mahanna]

  need to be careful here...
- Print statements. [Anthony Mahanna]
- Use `run_nx_tests` [Anthony Mahanna]
- Set env in ci. [Anthony Mahanna]
- Lint. [Anthony Mahanna]
- Pytest adopts typo. [Anthony Mahanna]
- Ci push branch. [Anthony Mahanna]

Other
~~~~~
- RTD Prep (#55) [Anthony Mahanna]

  * docs | wip

  * fix: `nx_to_nxadb`

  * fix: doc

  * checkpoint

  * checkpoint 2

  * fix: docstrings

  * checkpoint 3

  * fix: hyperlinks

  * mv: workflows
- Misc cleanup (#54) [Anthony Mahanna]

  * misc cleanup

  * fix: typo

  * fix: `test_shortest_path`
- Update readme (#50) [Anthony Mahanna]

  * update readme, initial commit

  * Update README.md

  * Update README.md

  * new: colab link

  * Update README.md

  * add video
- Update: `test_gpu` (#48) [Anthony Mahanna]

  * fix: `logger` instead of `print`

  * update `test_gpu_pagerank`

  * temp: remove gpu ci filter

  * remove: `Capturing`

  * add asserts

  * bring back filter

  * fix: import
- Cleanup `function.py` (#47) [Anthony Mahanna]

  * cleanup `function.py`

  * fix: typo, set `write_async` to False
- GA-163 | `test_multigraph` & `test_multidigraph` (#42) [Anthony
  Mahanna, hkernbach]

  * GA-163 | initial commit

  will fail

  * unlock adbnx

  * fix: `incoming_graph_data`

  * fix: incoming_graph_data

  * fix: off-by-one IDs

  * checkpoint

  * checkpoint: `BaseGraphTester` is passing

  * checkpoint: BaseGraphAttrTester

  * cleanup: `aql_fetch_data`, `aql_fetch_data_edge`

  * use pytest skip for failing tests

  * checkpoint: optimize `__iter__`

  * checkpoint: run `test_graph`

  * add comment

  * checkpoint

  * attempt: slleep

  * fix: lint

  * cleanup: getitem

  * cleanup: copy

  * attempt: shorten sleep

  * fix: `__set_adj_elements`

  * fix: mypy

  * attempt: decrease sleep

  * GA-163 | `test_digraph`

  * checkpoint

  lots of failures...

  * fix: set `self.Graph`

  * add type ignore

  * fix: graph name

  * fix: graph name

  * adjust assertions to exclude _rev, set `use_experimental_views`

  * Revert "adjust assertions to exclude _rev, set `use_experimental_views`"

  This reverts commit b8054192923915cb0769ef10bee9de41f7dc49ce.

  * fix: `_rev`, `use_experimental_views`

  * set `use_experimental_views`

  * fix: lint

  * new: `nbunch_iter` override

  * set experimental views to false

  * set experimental views to false

  * cleanup

  * GA-163 | `test_multigraph` checkpoint

  * fix lint

  * fix: `function.py`

  * cleanup: `graph`, `digraph`

  * fix: `test_data_input`

  * attempt: wait for CircleCI

  * fix: nx graph

  * remove sleep

  * new: `override` suffix

  * add override

  * enable more tests

  * fix: lint

  * checkpoint

  tests are still failing

  * checkpoint: 2 remaining test failures

  * fix: lint

  * checkpoint: one last failing test

  tried to debug this. no answer yet..

  * remove: `logger_debug`, fix lint

  * lint

  * fix: `test_multigraph`

  * cleanup, add missing test

  * new: `test_non_multigraph_input_a`

  * add comments

  * GA-163 | `test_multidigraph` (#45)

  * checkpoint: `test_multidigraph`

  * checkpoint: 1 failing test for each file: `test_digraph`, `test_multigraph`, `test_multidigraph`

  * fix: `test_to_undirected_reciprocal`

  * remove unused block

  * fix: `write_async` False

  ---------
- GA-163 | `test_digraph` (#40) [Anthony Mahanna]

  * GA-163 | initial commit

  will fail

  * unlock adbnx

  * fix: `incoming_graph_data`

  * fix: incoming_graph_data

  * fix: off-by-one IDs

  * checkpoint

  * checkpoint: `BaseGraphTester` is passing

  * checkpoint: BaseGraphAttrTester

  * cleanup: `aql_fetch_data`, `aql_fetch_data_edge`

  * use pytest skip for failing tests

  * checkpoint: optimize `__iter__`

  * checkpoint: run `test_graph`

  * add comment

  * checkpoint

  * attempt: slleep

  * fix: lint

  * cleanup: getitem

  * cleanup: copy

  * attempt: shorten sleep

  * fix: `__set_adj_elements`

  * fix: mypy

  * attempt: decrease sleep

  * GA-163 | `test_digraph`

  * checkpoint

  lots of failures...

  * fix: set `self.Graph`

  * add type ignore

  * fix: graph name

  * fix: graph name

  * adjust assertions to exclude _rev, set `use_experimental_views`

  * Revert "adjust assertions to exclude _rev, set `use_experimental_views`"

  This reverts commit b8054192923915cb0769ef10bee9de41f7dc49ce.

  * fix: `_rev`, `use_experimental_views`

  * set `use_experimental_views`

  * fix: lint

  * new: `nbunch_iter` override

  * set experimental views to false

  * set experimental views to false

  * cleanup

  * fix: `function.py`

  * cleanup: `graph`, `digraph`

  * fix: `test_data_input`

  * attempt: wait for CircleCI

  * fix: nx graph

  * remove sleep

  * new: `override` suffix

  * enable more tests

  * fix: lint
- GA-168 GPU Test (#43) [Anthony Mahanna, Heiko]

  * added gpu test using nx and cuda, first test commit for circleci - this is expected to fail

  * fix yml formatting

  * fix yml formatting again, define executr gpu

  * add test-gpu to matrix executor

  * fix resource class, added todo for later

  * flake8

  * pot deps fix

  * gpu test enable

  * gpu test enable

  * fix syntax

  * fix test, should work now on ci as  well

  * incr grid of graph

  * restructured test dirs, do not automatically run gpu tests.

  * isort

  * fmt, move test code

  * this is not allowed to be removed

  * fmt

  * test

  * 3.12 instead of 3.12.2 for gpu

  * new: `use_gpu` backend config

  * attempt: set `use_gpu`

  * force-set `use_gpu`

  * fix: lint

  * cleanup

  * fix: lint

  * fix imports

  * attempt: increase `digit`

  * new: `write_async` param

  * move assertions

  * fix lint

  ffs...

  * attempt: increase `digit`

  ---------
- GA-163 | `test_graph` (#33) [Anthony Mahanna]

  * GA-163 | initial commit

  will fail

  * unlock adbnx

  * fix: `incoming_graph_data`

  * fix: incoming_graph_data

  * fix: off-by-one IDs

  * checkpoint

  * checkpoint: `BaseGraphTester` is passing

  * checkpoint: BaseGraphAttrTester

  * cleanup: `aql_fetch_data`, `aql_fetch_data_edge`

  * use pytest skip for failing tests

  * checkpoint: optimize `__iter__`

  * checkpoint: run `test_graph`

  * add comment

  * checkpoint

  * attempt: slleep

  * fix: lint

  * cleanup: getitem

  * cleanup: copy

  * attempt: shorten sleep

  * fix: `__set_adj_elements`

  * fix: mypy

  * attempt: decrease sleep

  * fix: graph name

  * fix: `_rev`, `use_experimental_views`

  * new: `nbunch_iter` override

  * set experimental views to false

  * fix: lint
- [GA-153-2] AdjListInnerDict Update implementation (new) (#34) [Anthony
  Mahanna <43019056+aMahanna@users.noreply.github.com>    * Update
  nx_arangodb/classes/function.py    Co-authored-by: Anthony Mahanna
  <43019056+aMahanna@users.noreply.github.com>    * Update tests/test.py
  Co-authored-by: Anthony Mahanna
  <43019056+aMahanna@users.noreply.github.com>    * Update tests/test.py
  Co-authored-by: Anthony Mahanna
  <43019056+aMahanna@users.noreply.github.com>    * optimize
  separate_edges_by_collections    * fmt    * attempt to fix update in
  inner adj dict    * also test multigraphs    * fmt and mypy    * sort
  * remove obsolete comment    * use default node id    * fmt    *
  remove obsolete comment    * fix set adj elements by providing update
  * Update tests/test.py    Co-authored-by: Anthony Mahanna
  <43019056+aMahanna@users.noreply.github.com>    * Update tests/test.py
  Co-authored-by: Anthony Mahanna
  <43019056+aMahanna@users.noreply.github.com>    * fix test to new api
  ---------    Co-authored-by: Anthony Mahanna
  <43019056+aMahanna@users.noreply.github.com>, Heiko]

  * remove not needed imports, fix typos

  * moved over code from pr

  * fmt and lint

  * fix code, added test for graphs, added todo

  * adapt MultiGraph to old code

  * flake8

  * removed auto added import

  * add update method to CustomNodeView

  * update_local_nodes as private method

  * user logger instead of warnings

  * remove assertion, raise in case wrong key is given

  * move test only func into tst, removed unused func

  * remove import

  * TODO WIP

  * fix typo

  * move over code, will be broken as it is now

  * disabled this for now

  * fmt

  * fix mypy

  * py to 3.12

  * py to 3.12.3

  * py to 3.12.5

  * py to 3.12.5 ..............

  * back to 3.12.2

  * back to 3.10

  * fixes after merge

  * fix use of method

  * linting

  * make awesome linter happy

  * seriously.....

  * so wow

  * added core view

  * use proper class

  * Update nx_arangodb/classes/function.py
- Update to adj assertions, remove `_rev` concept (#37) [Anthony
  Mahanna]

  * initial commit

  * attempt: try to cache the update data

  * cleanup

  * update assertions

  * new: _rev assertions, `newDict` assertions

  this is currently failing on the `_rev` assertions for digraph & graph

  * Remove `_rev` concept (#39)

  * initial commit | remove `_rev` logic

  * remove: `root` concept

  * cleanup: `del "_rev"`

  * fix: lint

  * cleanup test

  * fix: return clause
- GA-169 | rename `graph_name` to `name` (#38) [Anthony Mahanna
  <43019056+aMahanna@users.noreply.github.com>    * do not allow graph
  renaming    * add test    * use setter in _set_graph_name, drop
  document before test execution    * fix nx tests, only supply g name
  if set    * add warning, assertion, and use protected    * priv to
  protected    ---------    Co-authored-by: Anthony Mahanna
  <43019056+aMahanna@users.noreply.github.com>, Anthony Mahanna, Heiko]

  * first commit graph name var to name only

  * Update nx_arangodb/classes/graph.py
- Nxadb_to_nx cleanup (#32) [Anthony Mahanna]

  * attempt: nxadb_to_nx cleanup

  * checkpoint

  * bring back other algorithms

  * passing, but certain assertions are commented out

  need to revisit failing assertions ASAP

  * attempt cleanup: nx overrides

  * cleanup: symmetrize_edges_if_directed

  * cleanup: `test_algorithm` assertions

  * fix: symmetrize edges

  * fix: symmetrize edges
- Use custom NodeView for `update` (#36) [Anthony Mahanna]
- [GA-153-1] Implement EdgeAttrDict update method (new) (#30) [Anthony
  Mahanna <43019056+aMahanna@users.noreply.github.com>    * Update
  nx_arangodb/classes/function.py    Co-authored-by: Anthony Mahanna
  <43019056+aMahanna@users.noreply.github.com>    * Update tests/test.py
  Co-authored-by: Anthony Mahanna
  <43019056+aMahanna@users.noreply.github.com>    * Update tests/test.py
  Co-authored-by: Anthony Mahanna
  <43019056+aMahanna@users.noreply.github.com>    * optimize
  separate_edges_by_collections    * fmt    ---------    Co-authored-by:
  Anthony Mahanna <43019056+aMahanna@users.noreply.github.com>, Heiko]

  * remove not needed imports, fix typos

  * moved over code from pr

  * fmt and lint

  * fix code, added test for graphs, added todo

  * adapt MultiGraph to old code

  * flake8

  * removed auto added import

  * add update method to CustomNodeView

  * update_local_nodes as private method

  * user logger instead of warnings

  * remove assertion, raise in case wrong key is given

  * move test only func into tst, removed unused func

  * remove import

  * TODO WIP

  * fix typo

  * disabled this for now

  * fix mypy

  * py to 3.12

  * py to 3.12.3

  * py to 3.12.5

  * py to 3.12.5 ..............

  * back to 3.12.2

  * back to 3.10

  * fixes after merge

  * fix use of method

  * added core view

  * Update nx_arangodb/classes/function.py
- Python Matrix (#35) [Anthony Mahanna]

  * initial commit

  * fix: machine

  * Update config.yml

  * try 3.12.2

  * fix: `-1` key in `EdgeKeyDict.__contains__`

  wow...
- [GA-153-0] Implement NodeDict update method  (#29) [Heiko]

  * remove not needed imports, fix typos

  * moved over code from pr

  * fmt and lint

  * add update method to CustomNodeView

  * update_local_nodes as private method

  * user logger instead of warnings

  * remove assertion, raise in case wrong key is given

  * move test only func into tst, removed unused func

  * remove import

  * fix typo

  * fix mypy

  * py to 3.12

  * py to 3.12.3

  * py to 3.12.5

  * py to 3.12.5 ..............

  * back to 3.12.2

  * back to 3.10
- Temp: lock adbnx. [Anthony Mahanna]
- Temp: return Any. [Anthony Mahanna]
- Add `phenolrs` as official dependency (#28) [Anthony Mahanna]

  * attempt: install `phenolrs` from test pypi

  * install `phenolrs` from regular pypi

  * add `phenolrs` to pyproject
- Added ability to load edge attrs. (#25) [Heiko]

  * added ability to load edge attrs.

  * also add to AdjListOuterDict

  * black fmt

  * fix init

  * updated phenolrs

  * remove now obsolete test

  * more tests, fixed load_all edge attr

  * added comment for clarity

  * better test name

  * move logic for edge attrs into one helper method, so it is only present in one location

  * fmt

  * import order

  * reformat msg, fix lang

  * applied suggested code changes

  * in fetch all for adjlist always load all edge attributes

  * add edge_values to coo representation and cache

  * fmt

  * fmt

  * remove not needed code anymore

  * added data definition for edge values

  * cleanup of unused imports

  * rm edge attrs of def args for adj
- GA-150 | MultiDiGraph Support (#26) [Anthony Mahanna]

  * GA-149 | initial commit

  failing for now

  * checkpoint

  no new tests yet, just experimenting with `AdjListInnerDict`

  * checkpoint 2

  * checkpoint 3

  still no new tests, just brainstorming

  * checkpoint 4

  starting to get messy...

  * cleanup & comments

  * comments

  * cleanup: `__contains__`

  * cleanup: `__getitem__`

  * restructuring

  * docstring updates

  * checkpoint 5

  * cleanup

  * new helper functions

  * checkpoint 6

  * checkpoint 7

  * cleanup

  * add warning

  * fix: conditional override

  * fix: func name

  * new: `FETCHED_ALL_IDS`

  Attribute used to establish if all ArangoDB IDs have been retrieved for the particular dict class. Not to be confused with `FETCHED_ALL_DATA`, which fetches both IDs & Documents

  * fix: parameterize `EDGE_TYPE_KEY`

  * cleanup: redundant code

  * fix: `nodes` & `edges` properties

  * new: `__process_int_edge_key`

  * new: `test_multigraph_*_crud`

  minimal suite for now. need to revisit

  * update: `test_algorithm` for `nxadb.MultiGraph`

  * fix: `__get_mirrored_adjlist_inner_dict`

  * extra docstring

  * new: graph overrides

  * fix: EdgeKeyDict docstring

  * update `phenolrs` wheel

  * fix: phenolrs

  * remove unused import

  * fix: except clause

  * fix: logger info

  * remove multigraph lock

  * fix: typo

  * cleanup: kwargs

  * remove print

  * fix: add `write_batch_size` to config

  this will be useful for bulk updates

  * temp: `NodeDict.update` hack

  Just a temporary solution. Will be removed shortly

  * revert ec1cbc8

  * add custom exception

  * update node & edge type logic for new vs existing graphs

  * fix: `symmetrize_edges` logic

  * GA-150 | initial commit
- GA-149 | MultiGraph Support (#20) [Anthony Mahanna]

  * GA-149 | initial commit

  failing for now

  * checkpoint

  no new tests yet, just experimenting with `AdjListInnerDict`

  * checkpoint 2

  * checkpoint 3

  still no new tests, just brainstorming

  * checkpoint 4

  starting to get messy...

  * cleanup & comments

  * comments

  * cleanup: `__contains__`

  * cleanup: `__getitem__`

  * restructuring

  * docstring updates

  * checkpoint 5

  * cleanup

  * new helper functions

  * checkpoint 6

  * checkpoint 7

  * cleanup

  * add warning

  * fix: conditional override

  * fix: func name

  * new: `FETCHED_ALL_IDS`

  Attribute used to establish if all ArangoDB IDs have been retrieved for the particular dict class. Not to be confused with `FETCHED_ALL_DATA`, which fetches both IDs & Documents

  * fix: parameterize `EDGE_TYPE_KEY`

  * cleanup: redundant code

  * fix: `nodes` & `edges` properties

  * new: `__process_int_edge_key`

  * new: `test_multigraph_*_crud`

  minimal suite for now. need to revisit

  * update: `test_algorithm` for `nxadb.MultiGraph`

  * fix: `__get_mirrored_adjlist_inner_dict`

  * extra docstring

  * new: graph overrides

  * fix: EdgeKeyDict docstring

  * update `phenolrs` wheel

  * fix: phenolrs

  * remove unused import

  * fix: except clause

  * fix: logger info

  * remove multigraph lock

  * fix: typo

  * cleanup: kwargs

  * remove print

  * fix: add `write_batch_size` to config

  this will be useful for bulk updates

  * temp: `NodeDict.update` hack

  Just a temporary solution. Will be removed shortly

  * revert ec1cbc8

  * add custom exception

  * update node & edge type logic for new vs existing graphs
- Async by default + data transfer cfg (#21) [Anthony Mahanna, Heiko]

  * async by default, cfg for batch size and parallelism level as parameter during graph init

  * lint

  * some param restructure - python convenient style now

  ---------
- [GA-161] prefix node and edge collections by graph name in case of
  named graph (#22) [Anthony Mahanna, Anthony Mahanna, Heiko]

  * prefox node and edge collections by graph name in case of named graph

  * undo changes in digraph.py

  * implement changes in main class and not in specific digraph class

  * rem newline

  ---------
- [GA-160] remove github action workflow, add circleci instead / Add
  .circleci/config.yml (#24) [Anthony Mahanna, Heiko]

  * Add .circleci/config.yml

  * removed github action

  * use docker machine supported executor for tests.

  * use ubuntu

  * debug list files

  * install wheel file by path name.

  * verbosity

  * explicitly use 3.10 python in machine executor

  * without update, add libssl and libffi deps

  * with update

  * try to use latest image

  * attempt to use pyenv.run for 3.10 python setup

  * fix wrong name during install apt get

  * Remove existing pyenv if exists

  * start arangodb first, install deps later (to allow arangodb to startup successfully). also try to use original command line command to install phenol wheel

  * use a different approach to install python version

  * auto find of wheel does not work. need to provide file path

  * 3.10 again

  * attempt: remove additional deps

  * remove `lint` requirement

  we can benefit from running them in parallel

  ---------
- GA-148 | DiGraph Support (#10) [Anthony Mahanna]

  * fix: protected instead of private

  * checkpoint

  * checkpoint 2

  * checkpoint

  * fix: traversal query

  * cleanup

  * fix: assertion

  * cleanup

  * update tests

  * cleanup

  * remove: `pull_graph` concept

  * update: `test_algorithm`

  * update assertions

  * cleanup: `_fetch_all()`

  * temp: `pull`

  * fix: `_fetch_all`

  * cleanup: `set_factory_methods`

  * remove: unused var

  * update `phenolrs` wheel

  * fix: lint

  * new: use `Enum`, cleanup `_fetch_all`

  * cleanup: docstrings
- [GA-157] Recursive GraphDict (#17) [Anthony Mahanna, Heiko]

  * moved tests, added root to G dict

  * all tests green

  * format, lint

  * fix a todo, fix flake

  * fix potential path that could be hit in case data structure is in unexpected state

  * use incr update instead

  * fixed missing parameter

  * added code suggestions, fixed update method in GraphDict which caused trouble

  * fix method signature

  * flake8

  * do not clear remote data if clear() is being called

  * fmt

  * GA-157 | review (#18)

  ---------
- GA-152 | Generalize Algorithm Dispatching (#11) [Anthony Mahanna]

  * GA-152 | initial commit

  * regen `_nx_arangodb`

  * remove: `backend_interface`

  not needed

  * revert 2573a75

  nevermind

  * cleanup

  * fix: `convert_to_nx`

  * temp: don't use `is_directed` & `is_multigraph`

  * update algorithm tests

  * cleanup

  * checkpoint

  this will failt until https://github.com/arangoml/phenolrs/pull/27/commits/b381686d44ff9a49d797caf7d79ea3749e758aed is built as a wheel file

  * update `phenolrs` wheel

  * cleanup

  * fix: imports & typing

  * update `phenolrs` wheel

  * comments

  * fix: lint

  * new: `number_of_edges` override

  * cleanup

  * fix: lint
- GA-156 | bump phenolrs (again) (#14) [Anthony Mahanna]
- GA-156 | update `phenolrs` wheel (#13) [Anthony Mahanna]

  * GA-156 | initial commit

  * fix: lint
- GA-154 | update `phenolrs` usage & use `nx.config` (#9) [Anthony
  Mahanna]

  * GA-147 | initial commit

  * new: recursive `EdgeAttrDict`

  * fix: `nested_keys` param

  * update tests

  * new: `AttrDict.root`

  * fix: `FETCHED_ALL_DATA`

  * checkpoint

  * checkpoint 2 (use NetworkX Config)

  * fix: lint

  * cleanup: `__fetch_all()`

  * fix: `self.clear()`

  * fix: `FETCHED_ALL_DATA` usage

  * fix: `logger_debug`

  * remove: walrus operator

  `:=` is acting weird... not sure what's going on

  * revert bccc1e6

  * new: `load_adj_dict_as_multigraph`

  * cleanup

  * fix: `logger_debug`
- GA-147 | recursive `NodeAttrDict` and `EdgeAttrDict` (#8) [Anthony
  Mahanna]

  * GA-147 | initial commit

  * new: recursive `EdgeAttrDict`

  * fix: `nested_keys` param

  * update tests

  * new: `AttrDict.root`

  * fix: `FETCHED_ALL_DATA`
- More housekeeping (#7) [Anthony Mahanna]

  * update gitignore

  * update password

  * add casting to string validation decorators

  * fix: decorator

  * cleanup: `nxadb.Graph.__init__`

  * update decorators

  * `pandas` as dev dep

  * new: `test_incoming_graph_data_not_nx_graph`

  * fix: `None` check

  * update `default_node_type`

  * cleanup
- Rename: `graph_exists` [Anthony Mahanna]
- Remove: `temp.py` [Anthony Mahanna]
- Update defaults. [Anthony Mahanna]
- Use kwarg. [Anthony Mahanna]
- Set defaults. [Anthony Mahanna]
- Cleanup: nxadb to nxcg. [Anthony Mahanna]
- More print statements. [Anthony Mahanna]
- Temp: print statements. [Anthony Mahanna]
- `nx.Graph` CRUD Interface (#3) [Anthony Mahanna]

  * cleanup: `DiGraph` & `Graph`

  * fix: `Digraph`

  * temp: hide `MultiGraph` & `MultiDiGraph`

  * checkpoint

  * new: `starter.sh` script for DB

  * skip test if missing `phenolrs`

  * checkpoint (again)

  last push before CI starts failing...

  * fix: `graph.py`

  * checkpoint (again)

  * bump

  * update tests

  * simplify `nx_arangodb` structure, update `dict.py`, cleanup

  * fix: use `orig_func`

  * checkpoint

  * remove multigraph

  will revisit later

  * update `_nx_arangodb`

  * new: `nx.shortest_path`

  * update tests

  * checkpoint (CI is failing)

  * remove duplicate file

  * fix: CI failure

  node removal was bugged

  * rename: `aql()` instead of `query()`

  * cleanup

  * HACK: `from_networkx_arangodb`

  need to revisit eventually

  * cleanup tests

  * fix: `logger` instead of `print`

  * checkpoint (again)

  * update: `test_edges_crud`

  * remove unused overrides

  * fix: aql functions

  * fix: address edge duplication for `nxadb.Graph`

  * add edge duplication test case

  * fix: typo

  * more `debug` logs :heart:

  * remove outdated comments

  * fix: debugs

  * fix: test typo

  * Update README.md

  * Update README.md

  * experimental: `CustomEdgeView`, `CustomEdgeDataView`

  * checkpoint

  * cleanup

  * update readme

  * update readme

  * new: `test_readme`

  * fix: bc

  * fix: shortest_path

  * add pass-through classes for `DiGraph`, `MultiGraph`, and `MultiDiGraph`

  * fix: `run_nx_tests`

  * cleanup

  * fix: `exceptions.py`

  * bump

  * fix: `create_using`

  * update readme

  * fix: nxcg

  * fix: type check

  * attempt fix: logger handler

  * attempt fix: logger
- Revert abfc928. [Anthony Mahanna]
- Update README.md. [Anthony Mahanna]
- Nxadb-to-nxcg (#2) [Anthony Mahanna]

  * wip: nxadb-to-nxcg

  using the adapter for now...

  * fix: typo

  * attempt fix: graph classes

  * fix: graph classes (again)

  * fix: typo

  * add DiGraph property

  not sure what's going on..

  * nxadb-to-nxcg (rust) | initial commit

  * print statements

  * fix: function name

  * fix: `as_directed`

  * more print statements

  * cleanup: `vertex_ids_to_index`

  * new: `parallelism` & `batch_size` kwargs

  hacky for now...

  * Update digraph.py

  * new: cache coo

  * cleanup

  * new: `louvain` & `pagerank`

  * fix: condition

  * update algorithms

  * cleanup

  * fix: bad import

  * cleanup: convert

  * new: Graph `pull` method

  * update `digraph`

  * fix: missing param

  * copy methods to digraph

  temporary workaround...

  * new: `load_adj_dict_as_undirected`
- Revert 634a762. [Anthony Mahanna]
- Cleanup: bc. [Anthony Mahanna]
- Update: bc. [Anthony Mahanna]
- Cleanup. [Anthony Mahanna]
- Update: test_bc. [Anthony Mahanna]
- Update: readme. [Anthony Mahanna]
- Remove: unused ci var. [Anthony Mahanna]
- Update: readme. [Anthony Mahanna]
- Update: pytest addopts. [Anthony Mahanna]
- Cleanup: ci. [Anthony Mahanna]
- Update: ci config. [Anthony Mahanna]
- Update: pytest adopts. [Anthony Mahanna]
- Initial commit. [Anthony Mahanna]


