#!/bin/bash

for i in $(find ./ -name '*.py');
do
  if ! grep -q License $i
  then
    cat LICENSE.md $i >$i.new && mv $i.new $i
  fi
done
