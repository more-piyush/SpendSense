<!--
  - CreateTransaction.vue
  - Copyright (c) 2019 james@firefly-iii.org
  -
  - This file is part of Firefly III (https://github.com/firefly-iii).
  -
  - This program is free software: you can redistribute it and/or modify
  - it under the terms of the GNU Affero General Public License as
  - published by the Free Software Foundation, either version 3 of the
  - License, or (at your option) any later version.
  -
  - This program is distributed in the hope that it will be useful,
  - but WITHOUT ANY WARRANTY; without even the implied warranty of
  - MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  - GNU Affero General Public License for more details.
  -
  - You should have received a copy of the GNU Affero General Public License
  - along with this program.  If not, see <https://www.gnu.org/licenses/>.
  -->

<template>
    <form accept-charset="UTF-8" class="form-horizontal" enctype="multipart/form-data">
        <input name="_token" type="hidden" value="xxx">
        <div v-if="error_message !== ''" class="row">
            <div class="col-lg-12">
                <div class="alert alert-danger alert-dismissible" role="alert">
                    <button class="close" data-dismiss="alert" type="button"
                            v-bind:aria-label="$t('firefly.close')"><span
                        aria-hidden="true">&times;</span></button>
                    <strong>{{ $t("firefly.flash_error") }}</strong> {{ error_message }}
                </div>
            </div>
        </div>

        <div v-if="success_message !== ''" class="row">
            <div class="col-lg-12">
                <div class="alert alert-success alert-dismissible" role="alert">
                    <button class="close" data-dismiss="alert" type="button"
                            v-bind:aria-label="$t('firefly.close')"><span
                        aria-hidden="true">&times;</span></button>
                    <strong>{{ $t("firefly.flash_success") }}</strong> <span v-html="success_message"></span>
                </div>
            </div>
        </div>
        <div>
            <div v-for="(transaction, index) in transactions" class="row">
                <div class="col-lg-12">
                    <div class="box">
                        <div class="box-header with-border">
                            <h3 class="box-title splitTitle">
                <span v-if="transactions.length > 1">{{ $t('firefly.single_split') }} {{ index + 1 }} / {{
                        transactions.length
                    }}</span>
                                <span v-if="transactions.length === 1">{{
                                        $t('firefly.transaction_journal_information')
                                    }}</span>
                            </h3>
                            <div v-if="transactions.length > 1" class="box-tools pull-right">
                                <button class="btn btn-xs btn-danger" type="button"
                                        v-on:click="deleteTransaction(index, $event)"><i
                                    class="fa fa-trash"></i></button>
                            </div>
                        </div>
                        <div class="box-body">
                            <div class="row">
                                <div id="transaction-info" class="col-lg-4">
                                    <transaction-description
                                        :value="transaction.description"
                                        :error="transaction.errors.description"
                                        :index="index"
                                        @input="updateDescription(index, $event)"
                                    >
                                    </transaction-description>
                                    <account-select
                                        :accountName="transaction.source_account.name"
                                        :accountTypeFilters="transaction.source_account.allowed_types"
                                        :defaultAccountTypeFilters="transaction.source_account.default_allowed_types"
                                        :error="transaction.errors.source_account"
                                        :index="index"
                                        :transactionType="transactionType"
                                        inputName="source[]"
                                        v-bind:inputDescription="$t('firefly.source_account')"
                                        v-on:clear:value="clearSource(index)"
                                        v-on:select:account="selectedSourceAccount(index, $event)"
                                    ></account-select>
                                    <account-select
                                        :accountName="transaction.destination_account.name"
                                        :accountTypeFilters="transaction.destination_account.allowed_types"
                                        :defaultAccountTypeFilters="transaction.destination_account.default_allowed_types"
                                        :error="transaction.errors.destination_account"
                                        :index="index"
                                        :transactionType="transactionType"
                                        inputName="destination[]"
                                        v-bind:inputDescription="$t('firefly.destination_account')"
                                        v-on:clear:value="clearDestination(index)"
                                        v-on:select:account="selectedDestinationAccount(index, $event)"
                                    ></account-select>
                                    <p v-if="0!== index && (null === transactionType || 'invalid' === transactionType || '' === transactionType)"
                                       class="text-warning">
                                        {{ $t('firefly.multi_account_warning_unknown') }}
                                    </p>
                                    <p v-if="0!== index && 'Withdrawal' === transactionType" class="text-warning">
                                        {{ $t('firefly.multi_account_warning_withdrawal') }}
                                    </p>
                                    <p v-if="0!== index && 'Deposit' === transactionType" class="text-warning">
                                        {{ $t('firefly.multi_account_warning_deposit') }}
                                    </p>
                                    <p v-if="0!== index && 'Transfer' === transactionType" class="text-warning">
                                        {{ $t('firefly.multi_account_warning_transfer') }}
                                    </p>
                                    <standard-date v-if="0===index"
                                                   v-model="transaction.date"
                                                   :error="transaction.errors.date"
                                                   :index="index"
                                    >
                                    </standard-date>
                                    <div v-if="index===0">
                                        <transaction-type
                                            :destination="transaction.destination_account.type"
                                            :source="transaction.source_account.type"
                                            v-on:set:transactionType="setTransactionType($event)"
                                            v-on:act:limitSourceType="limitSourceType($event)"
                                            v-on:act:limitDestinationType="limitDestinationType($event)"
                                        ></transaction-type>
                                    </div>
                                </div>
                                <div id="amount-info" class="col-lg-4">
                                    <amount
                                        :value="transaction.amount"
                                        :destination="transaction.destination_account"
                                        :error="transaction.errors.amount"
                                        :source="transaction.source_account"
                                        :transactionType="transactionType"
                                        @input="updateAmount(index, $event)"
                                    ></amount>
                                    <foreign-amount
                                        v-model="transaction.foreign_amount"
                                        :destination="transaction.destination_account"
                                        :error="transaction.errors.foreign_amount"
                                        :source="transaction.source_account"
                                        :transactionType="transactionType"
                                        v-bind:title="$t('form.foreign_amount')"
                                    ></foreign-amount>
                                </div>
                                <div id="optional-info" class="col-lg-4">
                                    <budget
                                        v-model="transaction.budget"
                                        :error="transaction.errors.budget_id"
                                        :no_budget="$t('firefly.none_in_select_list')"
                                        :transactionType="transactionType"
                                    ></budget>
                                    <category
                                        :value="transaction.category"
                                        :error="transaction.errors.category"
                                        :prediction-context="buildPredictionPayload(index)"
                                        :prediction-category="transaction.prediction.category"
                                        :prediction-confidence="transaction.prediction.confidence"
                                        :prediction-loading="transaction.prediction.loading"
                                        @input="updateCategory(index, $event)"
                                        @apply:prediction="applyPredictedCategory(index)"
                                    ></category>
                                    <piggy-bank
                                        v-model="transaction.piggy_bank"
                                        :error="transaction.errors.piggy_bank"
                                        :no_piggy_bank="$t('firefly.no_piggy_bank')"
                                        :transactionType="transactionType"
                                    ></piggy-bank>
                                    <tags
                                        v-model="transaction.tags"
                                        :error="transaction.errors.tags"
                                    ></tags>
                                    <bill
                                        v-model="transaction.bill"
                                        :error="transaction.errors.bill_id"
                                        :no_bill="$t('firefly.none_in_select_list')"
                                        :transactionType="transactionType"
                                    ></bill>
                                    <custom-transaction-fields
                                        v-model="transaction.custom_fields"
                                        :error="transaction.errors.custom_errors"
                                    ></custom-transaction-fields>
                                </div>
                            </div>
                        </div>
                        <div v-if="transactions.length-1 === index" class="box-footer">
                            <button class="split_add_btn btn btn-default" type="button" @click="addTransactionToArray">
                                {{ $t('firefly.add_another_split') }}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div v-if="transactions.length > 1" class="row">
            <div class="col-lg-6 col-md-6 col-sm-12 col-xs-12">
                <div class="box">
                    <div class="box-header with-border">
                        <h3 class="box-title">
                            {{ $t('firefly.split_transaction_title') }}
                        </h3>
                    </div>
                    <div class="box-body">
                        <group-description
                            v-model="group_title"
                            :error="group_title_errors"
                        ></group-description>
                    </div>
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-12 col-xs-12">
                <div class="box">
                    <div class="box-header with-border">
                        <h3 class="box-title">
                            {{ $t('firefly.submission') }}
                        </h3>
                    </div>
                    <div class="box-body">
                        <div class="checkbox">
                            <label>
                                <input v-model="createAnother" name="create_another" type="checkbox">
                                {{ $t('firefly.create_another') }}
                            </label>
                        </div>
                        <div class="checkbox">
                            <label v-bind:class="{ 'text-muted': this.createAnother === false}">
                                <input v-model="resetFormAfter" :disabled="this.createAnother === false"
                                       name="reset_form" type="checkbox">
                                {{ $t('firefly.reset_after') }}

                            </label>
                        </div>
                    </div>
                    <div class="box-footer">
                        <div class="btn-group">
                            <button id="submitButton" class="btn btn-success" @click="submit">{{
                                    $t('firefly.submit')
                                }}
                            </button>
                        </div>
                        <p class="text-success" v-html="success_message"></p>
                        <p class="text-danger" v-html="error_message"></p>
                    </div>
                </div>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-12 col-xs-12">
                <div class="box">
                    <div class="box-header with-border">
                        <h3 class="box-title">
                            {{ $t('firefly.submission_options') }}
                        </h3>
                    </div>
                    <div class="box-body">
                        <div class="checkbox">
                            <label>
                                <input v-model="applyRules" name="apply_rules" type="checkbox">
                                {{ $t('firefly.apply_rules_checkbox') }}
                            </label>
                        </div>
                        <div class="checkbox">
                            <label>
                                <input v-model="fireWebhooks" name="fire_webhooks" type="checkbox">
                                {{ $t('firefly.fire_webhooks_checkbox') }}

                            </label>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </form>
</template>

<script>
export default {
    name: "CreateTransaction",
    components: {},
    created() {
        this.addTransactionToArray();
        document.onreadystatechange = () => {
            if (document.readyState === "complete") {
                this.prefillSourceAccount();
                this.prefillDestinationAccount();
            }
        }
    },
    methods: {
        updateDescription(index, value) {
            this.transactions[index].description = value;
            this.scheduleCategoryPrediction(index);
        },
        updateAmount(index, value) {
            this.transactions[index].amount = value;
            this.scheduleCategoryPrediction(index);
        },
        updateCategory(index, value) {
            this.transactions[index].category = value;
        },
        applyPredictedCategory(index) {
            this.transactions[index].category = this.transactions[index].prediction.category;
            this.logCategoryFeedback(index, 'accepted', this.transactions[index].prediction.category);
        },
        prefillSourceAccount() {
            if (0 === window.sourceId) {
                return;
            }
            this.getAccount(window.sourceId, 'source_account');
        },
        prefillDestinationAccount() {
            if (0 === window.destinationId) {
                return;
            }
            this.getAccount(window.destinationId, 'destination_account');
        },
        getAccount(accountId, slot) {
            const uri = './api/v1/accounts/' + accountId + '?_token=' + document.head.querySelector('meta[name="csrf-token"]').content;
            axios.get(uri).then(response => {
                let model = response.data.data.attributes;
                model.type = this.fullAccountType(model.type, model.liability_type);
                model.id = parseInt(response.data.data.id);
                if ('source_account' === slot) {
                    this.selectedSourceAccount(0, model);
                }
                if ('destination_account' === slot) {
                    this.selectedDestinationAccount(0, model);
                }
            }).catch(error => {
                console.warn('Could  not auto fill account');
                console.warn(error);
            });

        },
        fullAccountType: function (shortType, liabilityType) {
            let searchType = shortType;
            if ('liabilities' === shortType) {
                searchType = liabilityType;
            }
            let arr = {
                'asset': 'Asset account',
                'loan': 'Loan',
                'debt': 'Debt',
                'mortgage': 'Mortgage'
            };
            return arr[searchType] ?? searchType;
        },
        normalizeAmount(value) {
            if ('string' !== typeof value) {
                return value;
            }
            if (1 === (value.match(/\,/g) || []).length) {
                return value.replace(',', '.');
            }
            return value;
        },
        getPredictionCurrency(transaction) {
            let transactionType = this.transactionType;
            if (!transactionType) {
                transactionType = '';
            }
            transactionType = transactionType.toLowerCase();

            if (['withdrawal', 'reconciliation', 'transfer'].includes(transactionType)) {
                return transaction.source_account.currency_code || null;
            }
            if ('deposit' === transactionType) {
                if (['debt', 'loan', 'mortgage'].includes((transaction.source_account.type || '').toLowerCase())) {
                    return transaction.source_account.currency_code || null;
                }
                return transaction.destination_account.currency_code || null;
            }
            return transaction.source_account.currency_code || transaction.destination_account.currency_code || null;
        },
        clearPrediction(index) {
            if ('undefined' === typeof this.transactions[index]) {
                return;
            }
            this.transactions[index].prediction = {
                category: '',
                confidence: null,
                loading: false,
                signature: '',
                transaction_id: '',
                model_family: '',
                model_version: '',
                predicted_categories: [],
                max_confidence: null,
                abstained: false,
                timestamp: '',
                inference_time_ms: null,
                feedback_history: {},
            };
        },
        buildFeedbackPayload(index, action, finalCategory = null) {
            const transaction = this.transactions[index];
            if ('undefined' === typeof transaction) {
                return null;
            }

            const prediction = transaction.prediction || {};
            if ('' === (prediction.category || '').trim()) {
                return null;
            }
            if ('' === (prediction.model_family || '').trim() || '' === (prediction.model_version || '').trim()) {
                return null;
            }

            const payload = {
                task: 'categorization',
                transaction_id: prediction.transaction_id || null,
                user_id: null,
                model_family: prediction.model_family,
                model_version: prediction.model_version,
                action: action,
                predicted_value: {
                    category: prediction.category,
                    confidence: prediction.confidence,
                    predicted_categories: prediction.predicted_categories || [],
                    max_confidence: prediction.max_confidence,
                    abstained: prediction.abstained,
                },
                final_value: null,
                metadata: {
                    source: 'firefly-ui',
                    description: transaction.description || '',
                    amount: this.normalizeAmount(transaction.amount || ''),
                    currency: this.getPredictionCurrency(transaction),
                    feedback_origin: 'create-transaction',
                },
                timestamp: new Date().toISOString(),
            };

            if (null !== finalCategory) {
                payload.final_value = {
                    category: finalCategory,
                };
            }

            return payload;
        },
        logCategoryFeedback(index, action, finalCategory = null) {
            const transaction = this.transactions[index];
            if ('undefined' === typeof transaction || 'undefined' === typeof window.categoryFeedbackUrl) {
                return Promise.resolve();
            }

            const payload = this.buildFeedbackPayload(index, action, finalCategory);
            if (null === payload) {
                return Promise.resolve();
            }

            const dedupeKey = JSON.stringify({
                action: action,
                predicted: payload.predicted_value.category,
                final: finalCategory,
            });
            if (transaction.prediction.feedback_history[dedupeKey]) {
                return Promise.resolve();
            }

            return axios.post(window.categoryFeedbackUrl, payload).then(() => {
                transaction.prediction.feedback_history[dedupeKey] = true;
            }).catch(error => {
                console.warn('Could not log SpendSense category feedback');
                console.warn(error);
            });
        },
        flushPredictionFeedback() {
            const feedbackRequests = [];

            this.transactions.forEach((transaction, index) => {
                const predictionCategory = ((transaction.prediction || {}).category || '').trim();
                if ('' === predictionCategory) {
                    return;
                }

                const finalCategory = (transaction.category || '').trim();
                if ('' === finalCategory) {
                    feedbackRequests.push(this.logCategoryFeedback(index, 'dismissed', null));
                    return;
                }

                if (finalCategory === predictionCategory) {
                    feedbackRequests.push(this.logCategoryFeedback(index, 'accepted', finalCategory));
                    return;
                }

                feedbackRequests.push(this.logCategoryFeedback(index, 'overridden', finalCategory));
            });

            return Promise.allSettled(feedbackRequests);
        },
        buildPredictionPayload(index) {
            const transaction = this.transactions[index];
            if ('undefined' === typeof transaction) {
                return null;
            }
            const description = (transaction.description || '').trim();
            const normalizedAmount = this.normalizeAmount(transaction.amount || '');
            const amount = parseFloat(normalizedAmount);

            if ('' === description || Number.isNaN(amount)) {
                return null;
            }

            return {
                description: description,
                amount: amount,
                currency: this.getPredictionCurrency(transaction),
                country: null,
            };
        },
        requestCategoryPrediction(index) {
            const payload = this.buildPredictionPayload(index);
            if (null === payload || 'undefined' === typeof window.categoryPredictionUrl) {
                this.clearPrediction(index);
                return;
            }

            const signature = JSON.stringify(payload);
            if (signature === this.transactions[index].prediction.signature) {
                return;
            }

            this.transactions[index].prediction.loading = true;
            axios.post(window.categoryPredictionUrl, payload).then(response => {
                const data = response.data || {};
                this.transactions[index].prediction.loading = false;
                this.transactions[index].prediction.signature = signature;
                this.transactions[index].prediction.category = data.suggested_category || '';
                this.transactions[index].prediction.confidence = data.confidence || data.max_confidence || null;
                this.transactions[index].prediction.transaction_id = data.transaction_id || '';
                this.transactions[index].prediction.model_family = data.model_family || '';
                this.transactions[index].prediction.model_version = data.model_version || '';
                this.transactions[index].prediction.predicted_categories = data.predicted_categories || [];
                this.transactions[index].prediction.max_confidence = data.max_confidence || null;
                this.transactions[index].prediction.abstained = !!data.abstained;
                this.transactions[index].prediction.timestamp = data.timestamp || '';
                this.transactions[index].prediction.inference_time_ms = data.inference_time_ms || null;
                this.transactions[index].prediction.feedback_history = {};

                // Make successful predictions visible even when the suggestion box is missed:
                // only auto-fill if the user has not already chosen a category.
                if ('' === (this.transactions[index].category || '').trim()
                    && '' !== this.transactions[index].prediction.category.trim()
                    && false === this.transactions[index].prediction.abstained) {
                    this.transactions[index].category = this.transactions[index].prediction.category;
                }
            }).catch(() => {
                this.transactions[index].prediction.loading = false;
                this.transactions[index].prediction.category = '';
                this.transactions[index].prediction.confidence = null;
            });
        },
        scheduleCategoryPrediction(index) {
            if ('undefined' !== typeof this.predictionTimers[index]) {
                clearTimeout(this.predictionTimers[index]);
            }
            this.predictionTimers[index] = setTimeout(() => {
                this.requestCategoryPrediction(index);
            }, 450);
        },
        convertData: function () {
            // console.log('Now in convertData()');
            let data = {
                'apply_rules': this.applyRules,
                'fire_webhooks': this.fireWebhooks,
                'transactions': [],
            };
            let transactionType;
            let firstSource;
            let firstDestination;

            if (this.transactions.length > 1) {
                data.group_title = this.group_title;
            }

            // get transaction type from first transaction
            transactionType = this.transactionType ? this.transactionType.toLowerCase() : 'invalid';

            // if the transaction type is invalid, might just be that we can deduce it from
            // the presence of a source or destination account
            firstSource = this.transactions[0].source_account.type;
            firstDestination = this.transactions[0].destination_account.type;
            // console.log('Type of first source is  ' + firstSource);

            if ('invalid' === transactionType && ['asset', 'Asset account', 'Loan', 'Debt', 'Mortgage'].includes(firstSource)) {
                transactionType = 'withdrawal';
            }

            if ('invalid' === transactionType && ['asset', 'Asset account', 'Loan', 'Debt', 'Mortgage'].includes(firstDestination)) {
                transactionType = 'deposit';
            }

            for (let key in this.transactions) {
                if (this.transactions.hasOwnProperty(key) && /^0$|^[1-9]\d*$/.test(key) && key <= 4294967294) {
                    data.transactions.push(this.convertDataRow(this.transactions[key], key, transactionType));
                }
            }

            // overrule group title in case its empty:
            if ('' === data.group_title && data.transactions.length > 1) {
                data.group_title = data.transactions[0].description;
            }

            return data;
        },
        convertDataRow(row, index, transactionType) {
            // console.log('Now in convertDataRow()');
            let tagList = [];
            let foreignAmount = null;
            let foreignCurrency = null;
            let currentArray;
            let sourceId;
            let sourceName;
            let destId;
            let destName;
            let date;

            sourceId = row.source_account.id;
            sourceName = row.source_account.name;
            destId = row.destination_account.id;
            destName = row.destination_account.name;

            date = row.date;
            if (index > 0) {
                date = this.transactions[0].date;
            }

            // if type is 'withdrawal' and destination is empty, cash withdrawal.
            if (transactionType === 'withdrawal' && '' === destName) {
                destId = window.cashAccountId;
            }

            // if type is 'deposit' and source is empty, cash deposit.
            if (transactionType === 'deposit' && '' === sourceName) {
                sourceId = window.cashAccountId;
            }

            // if index is over 0 and type is withdrawal or transfer, take source from index 0.
            if (index > 0 && (transactionType.toLowerCase() === 'withdrawal' || transactionType.toLowerCase() === 'transfer')) {
                sourceId = this.transactions[0].source_account.id;
                sourceName = this.transactions[0].source_account.name;
            }

            // if index is over 0 and type is deposit or transfer, take destination from index 0.
            if (index > 0 && (transactionType.toLowerCase() === 'deposit' || transactionType.toLowerCase() === 'transfer')) {
                destId = this.transactions[0].destination_account.id;
                destName = this.transactions[0].destination_account.name;
            }

            tagList = [];
            // loop tags
            for (let tagKey in row.tags) {
                if (row.tags.hasOwnProperty(tagKey) && /^0$|^[1-9]\d*$/.test(tagKey) && tagKey <= 4294967294) {
                    tagList.push(row.tags[tagKey].text);
                }
            }

            // set foreign currency info:
            if (row.foreign_amount.amount !== '' && parseFloat(row.foreign_amount.amount) !== .00) {
                foreignAmount = row.foreign_amount.amount;
                foreignCurrency = row.foreign_amount.currency_id;
            }
            if (foreignCurrency === row.currency_id) {
                foreignAmount = null;
                foreignCurrency = null;
            }

            // correct some id's
            if (0 === destId) {
                destId = null;
            }
            if (0 === sourceId) {
                sourceId = null;
            }
            // parse amount if has exactly one comma:
            // solves issues with some locales.
            if (1 === (row.amount.match(/\,/g) || []).length) {
                row.amount = row.amount.replace(',', '.');
            }

            currentArray =
                {
                    type: transactionType,
                    date: date,

                    amount: row.amount,
                    currency_id: row.currency_id,

                    description: row.description,

                    source_id: sourceId,
                    source_name: sourceName,

                    destination_id: destId,
                    destination_name: destName,


                    category_name: row.category,

                    interest_date: row.custom_fields.interest_date,
                    book_date: row.custom_fields.book_date,
                    process_date: row.custom_fields.process_date,
                    due_date: row.custom_fields.due_date,
                    payment_date: row.custom_fields.payment_date,
                    invoice_date: row.custom_fields.invoice_date,
                    internal_reference: row.custom_fields.internal_reference,
                    notes: row.custom_fields.notes,
                    external_url: row.custom_fields.external_url
                };

            if (tagList.length > 0) {
                currentArray.tags = tagList;
            }
            if (null !== foreignAmount) {
                currentArray.foreign_amount = foreignAmount;
                currentArray.foreign_currency_id = foreignCurrency;
            }
            // set budget id and piggy ID.
            if (parseInt(row.budget) > 0) {
                currentArray.budget_id = parseInt(row.budget);
            }
            if (parseInt(row.bill) > 0) {
                currentArray.bill_id = parseInt(row.bill);
            }
            if (parseInt(row.piggy_bank) > 0) {
                currentArray.piggy_bank_id = parseInt(row.piggy_bank);
            }
            return currentArray;
        },
        // submit transaction
        submit(e) {
            // console.log('Now in submit()');
            const uri = './api/v1/transactions?_token=' + document.head.querySelector('meta[name="csrf-token"]').content;
            const data = this.convertData();

            let button = $('#submitButton');
            button.prop("disabled", true);

            this.flushPredictionFeedback().finally(() => {
                axios.post(uri, data).then(response => {
                    // console.log('Did a successful POST');
                    // this method will ultimately send the user on (or not).
                    if (0 === this.collectAttachmentData(response)) {
                        // console.log('Will now go to redirectUser()');
                        this.redirectUser(response.data.data.id, response.data.data);
                    }
                }).catch(error => {
                    // give user errors things back.
                    // something something render errors.

                    console.error('Error in transaction submission.');
                    console.error(error);
                    this.parseErrors(error.response.data);

                    // something.
                    // console.log('enable button again.')
                    button.removeAttr('disabled');
                });
            });

            if (e) {
                e.preventDefault();
            }
        },
        escapeHTML(unsafeText) {
            let div = document.createElement('div');
            div.innerText = unsafeText;
            return div.innerHTML;
        },
        redirectUser(groupId, transactionData) {
            // console.log('In redirectUser()');
            // console.log(transactionData);
            let title = null === transactionData.attributes.group_title ? transactionData.attributes.transactions[0].description : transactionData.attributes.group_title;
            // console.log('Title is "' + title + '"');
            // if count is 0, send user onwards.
            if (this.createAnother) {
                // do message:
                this.success_message = this.$t('firefly.transaction_stored_link', {
                    ID: groupId,
                    title: this.escapeHTML(title)
                });
                this.error_message = '';
                if (this.resetFormAfter) {
                    // also clear form.
                    this.resetTransactions();
                    // do a short time out?
                    setTimeout(() => this.addTransactionToArray(), 50);
                    //this.addTransactionToArray();
                }

                // clear errors:
                this.setDefaultErrors();

                // console.log('enable button again.')
                let button = $('#submitButton');
                button.removeAttr('disabled');
            } else {
                //console.log('Will redirect to previous URL. (' + previousUrl + ')');
                window.location.href = window.previousUrl + '?transaction_group_id=' + groupId + '&message=created';
            }
        },

        collectAttachmentData(response) {
            // console.log('Now incollectAttachmentData()');
            let groupId = response.data.data.id;

            // reverse list of transactions?
            response.data.data.attributes.transactions = response.data.data.attributes.transactions.reverse();
            // array of all files to be uploaded:
            let toBeUploaded = [];

            // array with all file data.
            let fileData = [];

            // all attachments
            let attachments = $('input[name="attachments[]"]');

            // loop over all attachments, and add references to this array:
            for (const key in attachments) {
                if (attachments.hasOwnProperty(key) && /^0$|^[1-9]\d*$/.test(key) && key <= 4294967294) {
                    for (const fileKey in attachments[key].files) {
                        if (attachments[key].files.hasOwnProperty(fileKey) && /^0$|^[1-9]\d*$/.test(fileKey) && fileKey <= 4294967294) {
                            // include journal thing.
                            toBeUploaded.push(
                                {
                                    journal: response.data.data.attributes.transactions[key].transaction_journal_id,
                                    file: attachments[key].files[fileKey]
                                }
                            );
                        }
                    }
                }
            }
            let count = toBeUploaded.length;
            // console.log('Found ' + toBeUploaded.length + ' attachments.');

            // loop all uploads.
            for (const key in toBeUploaded) {
                if (toBeUploaded.hasOwnProperty(key) && /^0$|^[1-9]\d*$/.test(key) && key <= 4294967294) {
                    // create file reader thing that will read all of these uploads
                    (function (f, i, theParent) {
                        let fileReader = new FileReader();
                        fileReader.onloadend = function (evt) {
                            if (evt.target.readyState === FileReader.DONE) { // DONE == 2
                                fileData.push(
                                    {
                                        name: toBeUploaded[key].file.name,
                                        journal: toBeUploaded[key].journal,
                                        content: new Blob([evt.target.result])
                                    }
                                );
                                if (fileData.length === count) {
                                    theParent.uploadFiles(fileData, groupId, response.data.data);
                                }
                            }
                        };
                        fileReader.readAsArrayBuffer(f.file);
                    })(toBeUploaded[key], key, this);
                }
            }
            return count;
        },

        uploadFiles(fileData, groupId, transactionData) {
            let count = fileData.length;
            let uploads = 0;
            for (const key in fileData) {
                if (fileData.hasOwnProperty(key) && /^0$|^[1-9]\d*$/.test(key) && key <= 4294967294) {
                    // console.log('Creating attachment #' + key);
                    // axios thing, + then.
                    const uri = './api/v1/attachments';
                    const data = {
                        filename: fileData[key].name,
                        attachable_type: 'TransactionJournal',
                        attachable_id: fileData[key].journal,
                    };
                    axios.post(uri, data)
                        .then(response => {
                            // console.log('Created attachment #' + key);
                            // console.log('Uploading attachment #' + key);
                            const uploadUri = './api/v1/attachments/' + response.data.data.id + '/upload';
                            axios.post(uploadUri, fileData[key].content)
                                .then(attachmentResponse => {
                                    // console.log('Uploaded attachment #' + key);
                                    uploads++;
                                    if (uploads === count) {
                                        // finally we can redirect the user onwards.
                                        // console.log('FINAL UPLOAD');
                                        this.redirectUser(groupId, transactionData);
                                    }
                                    // console.log('Upload complete!');
                                    return true;
                                }).catch(error => {
                                console.error('[b] Could not upload');
                                console.error(error);
                                // console.log('Uploaded attachment #' + key);
                                uploads++;
                                if (uploads === count) {
                                    // finally we can redirect the user onwards.
                                    // console.log('FINAL UPLOAD');
                                    this.redirectUser(groupId, transactionData);
                                }
                                // console.log('Upload complete!');
                                return false;
                            });
                        }).catch(error => {
                        console.error('Could not create upload.');
                        console.error(error);
                        uploads++;
                        if (uploads === count) {
                            // finally we can redirect the user onwards.
                            // console.log('FINAL UPLOAD');
                            this.redirectUser(groupId, transactionData);
                        }
                        // console.log('Upload complete!');
                        return false;
                    });
                }
            }

        },


        setDefaultErrors: function () {
            for (const key in this.transactions) {
                if (this.transactions.hasOwnProperty(key) && /^0$|^[1-9]\d*$/.test(key) && key <= 4294967294) {
                    // console.log('Set default errors for key ' + key);
                    //this.transactions[key].description
                    this.transactions[key].errors = {
                        source_account: [],
                        destination_account: [],
                        description: [],
                        amount: [],
                        date: [],
                        budget_id: [],
                        bill_id: [],
                        foreign_amount: [],
                        category: [],
                        piggy_bank: [],
                        tags: [],
                        // custom fields:
                        custom_errors: {
                            interest_date: [],
                            book_date: [],
                            process_date: [],
                            due_date: [],
                            payment_date: [],
                            invoice_date: [],
                            internal_reference: [],
                            notes: [],
                            attachments: [],
                            external_url: [],
                        },
                    };
                }
            }
        },
        parseErrors: function (errors) {
            this.setDefaultErrors();
            this.error_message = "";
            if (typeof errors.errors === 'undefined') {
                this.success_message = '';
                this.error_message = errors.message;
            } else {
                this.success_message = '';
                this.error_message = this.$t('firefly.errors_submission');
            }
            let transactionIndex;
            let fieldName;

            for (const key in errors.errors) {
                if (errors.errors.hasOwnProperty(key)) {
                    if (key === 'group_title') {
                        this.group_title_errors = errors.errors[key];
                    }
                    if (key !== 'group_title') {
                        // lol dumbest way to explode "transactions.0.something" ever.
                        transactionIndex = parseInt(key.split('.')[1]);
                        fieldName = key.split('.')[2];
                        // set error in this object thing.
                        switch (fieldName) {
                            case 'amount':
                            case 'date':
                            case 'budget_id':
                            case 'bill_id':
                            case 'description':
                            case 'tags':
                                this.transactions[transactionIndex].errors[fieldName] = errors.errors[key];
                                break;
                            case 'type':
                                if(errors.errors[key].length > 0) {
                                    this.transactions[transactionIndex].errors.source_account = [this.$t('firefly.select_source_account')];
                                    this.transactions[transactionIndex].errors.destination_account = [this.$t('firefly.select_dest_account')];
                                }
                                break;
                            case 'source_name':
                            case 'source_id':
                                this.transactions[transactionIndex].errors.source_account =
                                    this.transactions[transactionIndex].errors.source_account.concat(errors.errors[key]);
                                break;
                            case 'destination_name':
                            case 'destination_id':
                                this.transactions[transactionIndex].errors.destination_account =
                                    this.transactions[transactionIndex].errors.destination_account.concat(errors.errors[key]);
                                break;
                            case 'foreign_amount':
                            case 'foreign_currency_id':
                                this.transactions[transactionIndex].errors.foreign_amount =
                                    this.transactions[transactionIndex].errors.foreign_amount.concat(errors.errors[key]);
                                break;
                        }
                    }
                    // unique some things
                    if (typeof this.transactions[transactionIndex] !== 'undefined') {
                        this.transactions[transactionIndex].errors.source_account =
                            Array.from(new Set(this.transactions[transactionIndex].errors.source_account));
                        this.transactions[transactionIndex].errors.destination_account =
                            Array.from(new Set(this.transactions[transactionIndex].errors.destination_account));
                    }

                }
            }
        },
        resetTransactions: function () {
            // console.log('Now in resetTransactions()');
            this.transactions = [];
            this.group_title = '';

        },
        addTransactionToArray: function (e) {
            // console.log('Now in addTransactionToArray()');
            this.transactions.push({
                description: "",
                date: "",
                amount: "",
                category: "",
                prediction: {
                    category: "",
                    confidence: null,
                    loading: false,
                    signature: "",
                    transaction_id: "",
                    model_family: "",
                    model_version: "",
                    predicted_categories: [],
                    max_confidence: null,
                    abstained: false,
                    timestamp: "",
                    inference_time_ms: null,
                    feedback_history: {},
                },
                piggy_bank: 0,
                errors: {
                    source_account: [],
                    destination_account: [],
                    description: [],
                    amount: [],
                    date: [],
                    budget_id: [],
                    bill_id: [],
                    foreign_amount: [],
                    category: [],
                    piggy_bank: [],
                    tags: [],
                    // custom fields:
                    custom_errors: {
                        interest_date: [],
                        book_date: [],
                        process_date: [],
                        due_date: [],
                        payment_date: [],
                        invoice_date: [],
                        internal_reference: [],
                        notes: [],
                        attachments: [],
                        external_url: [],
                    },
                },
                budget: 0,
                bill: 0,
                tags: [],
                custom_fields: {
                    "interest_date": "",
                    "book_date": "",
                    "process_date": "",
                    "due_date": "",
                    "payment_date": "",
                    "invoice_date": "",
                    "internal_reference": "",
                    "notes": "",
                    "attachments": [],
                    "external_url": "",
                },
                foreign_amount: {
                    amount: "",
                    currency_id: 0
                },
                source_account: {
                    id: 0,
                    name: "",
                    type: "",
                    currency_id: 0,
                    currency_name: '',
                    currency_code: '',
                    currency_decimal_places: 2,
                    allowed_types: ['Asset account', 'Revenue account', 'Loan', 'Debt', 'Mortgage'],
                    default_allowed_types: ['Asset account', 'Revenue account', 'Loan', 'Debt', 'Mortgage']
                },
                destination_account: {
                    id: 0,
                    name: "",
                    type: "",
                    currency_id: 0,
                    currency_name: '',
                    currency_code: '',
                    currency_decimal_places: 2,
                    allowed_types: ['Asset account', 'Expense account', 'Loan', 'Debt', 'Mortgage'],
                    default_allowed_types: ['Asset account', 'Expense account', 'Loan', 'Debt', 'Mortgage']
                }
            });
            if (this.transactions.length === 1) {
                // console.log('Length == 1, set date to today.');
                // set first date.
                let today = new Date();
                this.transactions[0].date = today.getFullYear() + '-' + ("0" + (today.getMonth() + 1)).slice(-2) + '-' + ("0" + today.getDate()).slice(-2)
                + 'T'+ ("0" + today.getHours()).slice(-2) +':' + ("0" + today.getMinutes()).slice(-2);
                //console.log(this.transactions[0].date);

                // call for extra clear thing:
                // this.clearSource(0);
                //this.clearDestination(0);
            }
            if (e) {
                e.preventDefault();
            }
        },
        setTransactionType: function (type) {
            this.transactionType = type;
            for (let i = 0; i < this.transactions.length; i++) {
                this.scheduleCategoryPrediction(i);
            }
        },

        deleteTransaction: function (index, event) {
            event.preventDefault();
            // console.log('Remove transaction.');
            this.transactions.splice(index, 1);
        },
        limitSourceType: function (type) {
            let i;
            for (i = 0; i < this.transactions.length; i++) {
                this.transactions[i].source_account.allowed_types = [type];
            }
        },
        limitDestinationType: function (type) {
            let i;
            for (i = 0; i < this.transactions.length; i++) {
                this.transactions[i].destination_account.allowed_types = [type];
            }
        },

        selectedSourceAccount: function (index, model) {
            // console.log('Now in selectedSourceAccount()');
            if (typeof model === 'string') {
                //console.log('model is string.')
                // cant change types, only name.
                this.transactions[index].source_account.name = model;
            } else {
                //console.log('model is NOT string.')
                this.transactions[index].source_account = {
                    id: model.id,
                    name: model.name,
                    type: model.type,
                    currency_id: model.currency_id,
                    currency_name: model.currency_name,
                    currency_code: model.currency_code,
                    currency_decimal_places: model.currency_decimal_places,
                    allowed_types: this.transactions[index].source_account.allowed_types,
                    default_allowed_types: ['Asset account', 'Revenue account', 'Loan', 'Debt', 'Mortgage']
                };
                if(model.hasOwnProperty('account_currency_id') && null !== model.account_currency_id) {
                    this.transactions[index].source_account.currency_id = model.account_currency_id;
                    this.transactions[index].source_account.currency_name = model.account_currency_name;
                    this.transactions[index].source_account.currency_code = model.account_currency_code;
                    this.transactions[index].source_account.currency_decimal_places = model.account_currency_decimal_places;
                }

                // force types on destination selector.
                this.transactions[index].destination_account.allowed_types = window.allowedOpposingTypes.source[model.type];
            }
            this.scheduleCategoryPrediction(index);
            //console.log('Transactions:');
            //console.log(this.transactions);
        },
        selectedDestinationAccount: function (index, model) {
            // console.log('Now in selectedDestinationAccount()');
            if (typeof model === 'string') {
                // cant change types, only name.
                this.transactions[index].destination_account.name = model;
            } else {
                this.transactions[index].destination_account = {
                    id: model.id,
                    name: model.name,
                    type: model.type,
                    currency_id: model.currency_id,
                    currency_name: model.currency_name,
                    currency_code: model.currency_code,
                    currency_decimal_places: model.currency_decimal_places,
                    allowed_types: this.transactions[index].destination_account.allowed_types,
                    default_allowed_types: ['Asset account', 'Expense account', 'Loan', 'Debt', 'Mortgage']
                };
                if(model.hasOwnProperty('account_currency_id') && null !== model.account_currency_id) {
                    this.transactions[index].destination_account.currency_id = model.account_currency_id;
                    this.transactions[index].destination_account.currency_name = model.account_currency_name;
                    this.transactions[index].destination_account.currency_code = model.account_currency_code;
                    this.transactions[index].destination_account.currency_decimal_places = model.account_currency_decimal_places;
                }

                // force types on destination selector.
                this.transactions[index].source_account.allowed_types = window.allowedOpposingTypes.destination[model.type];
            }
            this.scheduleCategoryPrediction(index);
        },
        clearSource: function (index) {
            // console.log('clearSource(' + index + ')');
            // reset source account:
            this.transactions[index].source_account = {
                id: 0,
                name: '',
                type: '',
                currency_id: 0,
                currency_name: '',
                currency_code: '',
                currency_decimal_places: 2,
                allowed_types: this.transactions[index].source_account.allowed_types,
                default_allowed_types: ['Asset account', 'Revenue account', 'Loan', 'Debt', 'Mortgage']
            };
            // reset destination allowed account types.
            this.transactions[index].destination_account.allowed_types = [];

            // if there is a destination model, reset the types of the source
            // by pretending we selected it again.
            if (this.transactions[index].destination_account) {
                this.selectedDestinationAccount(index, this.transactions[index].destination_account);
            }
            this.scheduleCategoryPrediction(index);
        },
        clearDestination: function (index) {
            // console.log('clearDestination(' + index + ')');
            // reset destination account:
            this.transactions[index].destination_account = {
                id: 0,
                name: '',
                type: '',
                currency_id: 0,
                currency_name: '',
                currency_code: '',
                currency_decimal_places: 2,
                allowed_types: this.transactions[index].destination_account.allowed_types,
                default_allowed_types: ['Asset account', 'Expense account', 'Loan', 'Debt', 'Mortgage']
            };
            // reset destination allowed account types.
            this.transactions[index].source_account.allowed_types = [];

            // if there is a source model, reset the types of the destination
            // by pretending we selected it again.
            if (this.transactions[index].source_account) {
                this.selectedSourceAccount(index, this.transactions[index].source_account);
            }
            this.scheduleCategoryPrediction(index);
        }
    },

    /*
     * The component's data.
     */
    data() {
        return {
            transactionType: null,
            group_title: "",
            transactions: [],
            group_title_errors: [],
            error_message: "",
            success_message: "",
            cash_account_id: 0,
            createAnother: false,
            resetFormAfter: false,
            fireWebhooks: true,
            applyRules: true,
            resetButtonDisabled: true,
            attachmentCount: 0,
            predictionTimers: {},
        };
    },
}
</script>
